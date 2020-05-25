import argparse
import pathlib
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import tqdm
from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument("base_dir")
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')


def get_image_paths(base_dir: pathlib.Path):
    return list(base_dir.joinpath("images").glob("*.png"))


def center_crop(image):
    
    if image.shape[0] == image.shape[1]:
        return image
    assert image.shape[1] > image.shape[0]
    missing = (image.shape[1] - image.shape[0]) // 2
    return image[:, missing:-missing]

if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    args, unknown = parser.parse_known_args()
    model = InpaintCAModel()
    base_dir = pathlib.Path(args.base_dir)
    image_paths = get_image_paths(base_dir)
    sess_config = tf.ConfigProto()
    images_all = [cv2.imread(str(impath)) for impath in image_paths]
    masks_all = [cv2.imread(str(base_dir.joinpath("masks").joinpath(impath.name))) for impath in image_paths]
    imshape = cv2.imread(str(image_paths[0])).shape
    sess_config.gpu_options.allow_growth = True
    input_image = tf.placeholder(tf.float32, [1, 256, 512, 3])
    with tf.Session(config=sess_config) as sess:
#        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(FLAGS, input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')

        for idx_, impath in enumerate(tqdm.tqdm(image_paths)):
            image = images_all[idx_]
            mask_path = base_dir.joinpath("masks").joinpath(impath.name)
            mask = 255 - masks_all[idx_] 
#            image = center_crop(image)
#            mask = center_crop(mask)
#            image = cv2.resize(image, (256,256))
#            mask = cv2.resize(mask, (256,256))

            assert image.shape == mask.shape

#            h, w, _ = image.shape
#            grid = 8
#            image = image[:h//grid*grid, :w//grid*grid, :]
#            mask = mask[:h//grid*grid, :w//grid*grid, :]

            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            inp_img = np.concatenate([image, mask], axis=2)

            result = sess.run(output, feed_dict={input_image: inp_img})
            target_path = base_dir.joinpath("gated_conv").joinpath(impath.name)
            target_path.parent.mkdir(exist_ok=True)
#            cv2.imwrite(str(target_path), result[0][:, :, ::-1])
#            print("Saved to:", target_path)
