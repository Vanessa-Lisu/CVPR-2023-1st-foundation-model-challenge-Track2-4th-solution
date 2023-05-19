from run_beit3_finetuning import get_args
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import json
import os
from timm.models import create_model
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, \
    LayerDecayValueAssigner, get_is_head_flag_for_vit

from engine_for_finetuning import get_handler, evaluate
from datasets import create_downstream_dataset
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from data_utils import read_image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_images_texts(images_root, texts_root, test=False):
    images = []
    texts = []
    f = open(texts_root, 'r')
    for line in f.readlines():
        text = line.strip()
        texts.append(text)
    idx_dic = {}
    cnts = 0
    images_names = os.listdir(images_root)
    images_names.sort()
    for image_name in images_names:
        img = read_image(os.path.join(images_root, image_name), "RGB")
        # img = transform_ops(img)
        images.append(img)
        idx_dic[cnts] = image_name
        cnts += 1
    if test:
        images = images[:20]
        texts = texts[:20]
    return images, texts, idx_dic


def infer(similarity, idx_dic, texts, category, num_car):
    # similarity = similarity.detach.cpu()
    similarity_argsort = np.argsort(-similarity, axis=1)
    print('similarity_argsort.shape', similarity_argsort.shape)

    topk = 10
    result_list = []
    for i in range(len(similarity_argsort)):
        dic = {'text': texts[i], 'image_names': []}
        for j in range(topk):
            if(category=='car'):    #car model
                if i < num_car:     #car
                    dic['image_names'].append(idx_dic[similarity_argsort[i,j]]) #正序
                else:
                    dic['image_names'].append(idx_dic[similarity_argsort[i,-j-1]])
            else:   #person model
                if i < num_car: #car
                    dic['image_names'].append(idx_dic[similarity_argsort[i,-j-1]])
                else:
                    dic['image_names'].append(idx_dic[similarity_argsort[i,j]]) #正序

            # if category in idx_dic[similarity_argsort[i,j]]:
            #     dic['image_names'].append(idx_dic[similarity_argsort[i,j]])
            # else:
            #     dic['image_names'].append(idx_dic[similarity_argsort[i,len(similarity_argsort)-j-1]])
        result_list.append(dic)
    with open('infer_json_{}.json'.format(category), 'w') as f:
        f.write(json.dumps({'results': result_list}, indent=4))


def main(args, ds_init, category):

    if ds_init is not None:
        utils.create_ds_config(args)

    if args.task_cache_path is None:
        args.task_cache_path = args.output_dir

    # print('args',args)

    device = torch.device(args.device)

    cudnn.benchmark = True

    if utils.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if not args.model.endswith(args.task):
        if args.task in ("flickr30k", "coco_retrieval"):
            model_config = "%s_retrieval" % args.model
        elif args.task in ("coco_captioning", "nocaps"):
            model_config = "%s_captioning" % args.model
        elif args.task in ("imagenet"):
            model_config = "%s_imageclassification" % args.model
        else:
            model_config = "%s_%s" % (args.model, args.task)
    else:
        model_config = args.model
    print("model_config = %s" % model_config)
    model = create_model(
        model_config,
        pretrained=False,
        drop_path_rate=args.drop_path,
        vocab_size=args.vocab_size,
        checkpoint_activations=args.checkpoint_activations,
    )

    if category == 'car':
        utils.load_model_and_may_interpolate(args.finetune_car, model, args.model_key, args.model_prefix)
    if category == 'people':
        utils.load_model_and_may_interpolate(args.finetune_people, model, args.model_key, args.model_prefix)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    #print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        lrs = list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
        assigner = LayerDecayValueAssigner(lrs)
    elif args.task_head_lr_weight > 1:
        assigner = LayerDecayValueAssigner([1.0, args.task_head_lr_weight], scale_handler=get_is_head_flag_for_vit)
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()

    # if args.distributed:
    #     torch.distributed.barrier()
    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        # if args.distributed:
        #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        #     model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    task_handler = get_handler(args)
    data_loader_test = create_downstream_dataset(args, is_eval=True)
    if args.task in ["nlvr2", "flickr30k", "coco_retrieval", "imagenet"]:
        ext_test_stats, task_key, similarity = evaluate(data_loader_test, model, device, task_handler)
        print("get similarity")
        return similarity.cpu().detach().numpy()


def get_contents(root):
    with open(root,'r') as f:
        contents = json.load(f)
    print(len(contents.get('results')))
    return contents.get('results')


def concate(car_json_root, people_json_root, num_car):
    car_result = get_contents(car_json_root)
    people_result = get_contents(people_json_root)
    all_num = len(car_result)
    print('all_num', all_num)
    all_results = []
    for i in range(num_car):
        car = {'text' : car_result[i].get('text') , 'image_names':[]}
        for j in range(10):
            car['image_names'].append(car_result[i].get('image_names')[j])
        all_results.append(car)

    for k in range(num_car,all_num):
        people = {'text' : people_result[k].get('text') , 'image_names':[]}
        for kk in range(10):
            people['image_names'].append(people_result[k].get('image_names')[kk])
        all_results.append(people)

    with open('infer_json_all.json', 'w') as f:
        f.write(json.dumps({'results': all_results}, indent=4))


def get_num_car_txt(txt_path):
    
    car = []
    people = []
    num_car = 0
    with open(txt_path) as f:
        for line in f:
            if 'He ' in line or 'She ' in line:
                people.append(line)
            else:
                car.append(line)
                num_car += 1
   
    return num_car


if __name__ == '__main__':
    images_root = '/home/aistudio/dataset/test/test_images'
    texts_root = '/home/aistudio/dataset/test/test.txt'
    
    num_car = orderTxt(texts_root)

    opts, ds_init = get_args()   
    category = ['car', 'people']

    num_car = get_num_car_txt(texts_root)
    print(num_car)

    images, texts, idx_dic = get_images_texts(images_root, texts_root, test=False)
    
    # 分别对人和车进行推理
    for i in range(len(category)):
        opts.category = category[i]
        print(opts.category)
        similarity = main(opts, ds_init, category[i])
        print("start infer")
        infer(similarity, idx_dic, texts, category[i], num_car)
    
    # 将人和车的infer文件进行拼接
    car_json_root = "./infer_json_car.json"
    people_json_root = "./infer_json_people.json"
    
    # concate(car_json_root, people_json_root, get_num_car_txt(texts_root))
    concate(car_json_root, people_json_root, num_car)
