"""A binary for training depth and egomotion."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
from absl import app
from absl import logging
import json
import sys

import depth_motion_field_model
from parameter_container import ParameterContainer
import cv2
import numpy as np
import tensorflow as tf



TRAINER_PARAMS = {
    # Learning rate
    'learning_rate': 2e-4,

    # If not None, gradients will be clipped to this value.
    'clip_gradients': 10.0,

    # Number of iterations in the TPU internal on-device loop.
    'iterations_per_loop': 20,

    # If not None, the training will be initialized form this checkpoint.
    'init_ckpt': None,

    # A string, specifies the format of a checkpoint form which to initialize.
    # The model code is expected to convert this string into a
    # vars_to_restore_fn (see below),
    'init_ckpt_type': None,

    # Master address
    'master': None,

    # Directory where checkpoints will be saved.
    'model_dir': None,

    # Maximum number of training steps.
    'max_steps': int(1e6),

    # Number of hours between each checkpoint to be saved.
    # The default value of 10,000 hours effectively disables the feature.
    'keep_checkpoint_every_n_hours': 10000,
}

class InitFromCheckpointHook(tf.estimator.SessionRunHook):
    """A hook for initializing training from a checkpoint.

  Although the Estimator framework supports initialization from a checkpoint via
  https://www.tensorflow.org/api_docs/python/tf/estimator/WarmStartSettings,
  the only way to build mapping between the variables and the checkpoint names
  is via providing a regex. This class provides the same functionality, but the
  mapping can be built by a callback, which provides more flexibility and
  readability.
  """

    def __init__(self, model_dir, ckpt_to_init_from, vars_to_restore_fn=None):
        """Creates an instance.

    Args:
      model_dir: A string, path where checkpoints are saved during training.
        Used for checking whether a checkpoint already exists there, in which
        case we want to continue training from there rather than initialize from
        another checkpoint.
      ckpt_to_init_from: A string, path to a checkpoint to initialize from.
      vars_to_restore_fn: A callable that receives no arguments. When called,
        expected to provide a dictionary that maps the checkpoint name of each
        variable to the respective variable object. This dictionary will be used
        as `var_list` in a Saver object used for initializing from
        `ckpt_to_init_from`. If None, the default saver will be used.
    """
        self._ckpt = None if tf.train.latest_checkpoint(
            model_dir) else ckpt_to_init_from
        self._vars_to_restore_fn = vars_to_restore_fn

    def begin(self):
        if not self._ckpt:
            return
        logging.info('%s will be used for initialization.', self._ckpt)
        # Build a saver object for initializing from a checkpoint, or use the
        # default one if no vars_to_restore_fn was given.
        self._reset_step = None
        if tf.train.get_global_step() is not None:
            self._reset_step = tf.train.get_global_step().assign(0)
        if not self._vars_to_restore_fn:
            logging.info('All variables will be initialized form the checkpoint.')
            self._saver = tf.get_collection(tf.GraphKeys.SAVERS)[0]
            return

        vars_to_restore = self._vars_to_restore_fn()
        restored_vars_string = (
            'The following variables are to be initialized from the checkpoint:\n')
        for ckpt_name in sorted(vars_to_restore):
            restored_vars_string += '%s --> %s\n' % (
                ckpt_name, vars_to_restore[ckpt_name].op.name)

        logging.info(restored_vars_string)
        self._saver = tf.train.Saver(vars_to_restore)

    def after_create_session(self, session, coord):
        del coord  # unused
        if not self._ckpt:
            return
        self._saver.restore(session, self._ckpt)
        self._saver.restore(session, self._ckpt)
        if self._reset_step is not None:
            session.run(self._reset_step)


class InferDepthMotion():
    def __init__(self, model_dir):
        self.params = ParameterContainer({
            'model': {
                'batch_size': 1,
                'input': {}
            },
        }, {'trainer': {
            'master': None,
            'model_dir': model_dir
        }})

        get_vars_to_restore_fn = depth_motion_field_model.get_vars_to_restore_fn
        init_ckpt_type = self.params.trainer.get('init_ckpt_type')
        vars_to_restore_fn = (get_vars_to_restore_fn(init_ckpt_type) if init_ckpt_type else None)
        trainer_params = ParameterContainer.from_defaults_and_overrides(TRAINER_PARAMS, self.params.trainer, is_strict=True)

        run_config_params = {
            'model_dir':
                trainer_params.model_dir,
            'save_summary_steps':
                5,
            'keep_checkpoint_every_n_hours':
                trainer_params.keep_checkpoint_every_n_hours,
            'log_step_count_steps':
                25,
        }
        logging.info(
            'Estimators run config parameters:\n%s',
            json.dumps(run_config_params, indent=2, sort_keys=True, default=str))
        run_config = tf.estimator.RunConfig(**run_config_params)


        self.init_hook = InitFromCheckpointHook(trainer_params.model_dir,
                                        trainer_params.init_ckpt,
                                        vars_to_restore_fn)

        self.estimator = tf.estimator.Estimator(
            model_fn=self.estimator_spec_fn_infer,
            config=run_config,
            params=self.params.model.as_dict())

    def input_fn_infer(self,input_image):
        return tf.estimator.inputs.numpy_input_fn(x={"rgb": input_image}, num_epochs=1, shuffle=False)


    def estimator_spec_fn_infer(self,features, labels, mode, params):
        del labels # unused
        # depth estimation output of network
        depth_net_out = depth_motion_field_model.infer_depth(rgb_image=features['rgb'], params=params)

        return(tf.estimator.EstimatorSpec(mode=mode, predictions=depth_net_out))


    def run_local_inference(self, input_fn, size):
        """Run a simple single-mechine traing loop.

    Args:
            input_fn: A callable that complies with tf.Estimtor's definition of
        input_fn.
        trainer_params_overrides: A dictionary or a ParameterContainer with
        overrides for the default values in TRAINER_PARAMS above.
        model_params: A ParameterContainer that will be passed to the model (i. e.
        to losses_fn and input_fn).
        vars_to_restore_fn: A callable that receives no arguments. When called,
        expected to provide a dictionary that maps the checkpoint name of each
        variable to the respective variable object. This dictionary will be used
        as `var_list` in a Saver object used for initializing from the checkpoint
        at trainer_params.init_ckpt. If None, the default saver will be used.
    """

        predict_output = self.estimator.predict(input_fn=input_fn, predict_keys=None, hooks=[self.init_hook])
        predict_output = np.array(list(predict_output))

        
        depth = predict_output[0, :, :, :]
        depth = cv2.resize(depth, (size[0], size[1]))
    
        return depth


    def disp_to_depth(self, disp, min_depth, max_depth):
        min_disp = 1 / max_depth 
        max_disp = 1 / min_depth 
        scaled_disp = min_disp + (max_disp - min_disp) * disp 
        depth = 1 / scaled_disp
        return scaled_disp, depth

    def draw_depth_map(self,image, disp, filename,output_dir):
        cv2.imwrite(os.path.join(output_dir,filename+'_raw.png'),image)
        img_path = os.path.join(output_dir,filename+'_depth.png')

        import matplotlib.pyplot as plt
        # Visualization method 1
        depth = 1 / disp[:, :]
        pred_ = cv2.resize(depth, (image.shape[1],image.shape[0]))
        plt.imsave(img_path, pred_, cmap='plasma')

        # # Visualization method 2
        # scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
        # vmax = np.percentile(scaled_disp, 95)
        # plt.imsave(img_path, scaled_disp, cmap='magma', vmax=vmax)

    def infer(self, input_fn,size):
        """Run inference.

    Args:
        input_fn: A tf.Estimator compliant input_fn.
            get_vars_to_restore_fn: A callable that receives a string argument
        (intdicating the type of initialization) and returns a vars_to_restore_fn.
        The latter is a callable that receives no arguments and returns a
        dictionary that can be passed to a tf.train.Saver object's constructor as
        a `var_list` to indicate which variables to load from what names in the
        checnpoint.
    """

        depth = self.run_local_inference(input_fn,size)
        return depth


def main():
    model_dir = '/home/tata/Project/depth_and_motion_learning/log/kitti_experiment'
    depmo_model = InferDepthMotion(model_dir)

    width,height = 416,128
    data_dir = '/home/tata/Project/dataset/kitti_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data'
    output_dir = '/home/tata/Project/depth_and_motion_learning/output/test_kitti'
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    file_list = sorted(os.listdir(data_dir))
  
    for ii,file in enumerate(file_list):
        print(ii)
        file = file.split()[0]
        image_path = os.path.join(data_dir, file)
        filename = file.split('/')[-1].split('.')[0] 
        input_image = cv2.imread(image_path).astype(np.float32)
        input_image = cv2.resize(input_image,(width, height)) 
        input_batch = np.reshape(input_image, (1, height, width, 3))

        depth = depmo_model.infer(depmo_model.input_fn_infer(input_image=input_batch), (width, height))

        depmo_model.draw_depth_map(input_image, depth, filename, output_dir)



if __name__ == '__main__':
    main()