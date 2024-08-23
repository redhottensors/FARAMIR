# FARAMIR - Fantasy Aesthetic Realism Assessment Model with Integrated Relabeling
FARAMIR is an ensemble-of-experts image content and quality assessment model for 2D and 3D artwork.

The model is trained to produce a binary classification output of "accept" or "reject" and an
additional label indicating the primary reason for the assessment. The decision threshold can be set
to any number between 0 and 1 to trade off precision and recall. The model defaults to a threshold
of 0.5 and generally performs "best" at this threshold.

## Installation and Usage
```sh
git clone --depth=1 https://github.com/redhottensors/FARAMIR.git
cd FARAMIR
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

source venv/bin/activate
python inference.py --help
```
Tested on python 3.11 and 3.12 with pytorch 2.4.0 CUDA, timm 1.0.8, and numpy 1.26.4. The
distributed weights are compatible with ``torch.load(weights_only=True)``.

## Class Labels
- Accept-Biased
  #. ``3d_realism`` -- A realistic 3D model without a prominent human.
  #. ``3d_human`` -- A semi-realistic human 3D model.
  #. ``3d_scene`` -- A realistic 3D scene, potentially involving action or multiple interacting
     subjects. May contain moderate violence.
  #. ``3d_environment`` -- An idealized 3D scene, usually without significant subjects, and with
     pleasing aesthetics.

  #. ``3d_untex`` -- An untextured 3D model of similar overall quality and content to ``3d_realism``.
  #. ``3d_untex_human`` -- An untextured 3D model of similar overall quality and content to
     ``3d_human``.

  #. ``2d_fantasy`` -- A quality 2D fantasy illustration, generally not invoking a prominent human
     subject.
  #. ``2d_human`` -- A 2D depiction of a human of generally good quality.
  #. ``2d_anime`` -- A 2D depiction of a human with traditional anime characteristics. Unfortunately
     and understandably rather female-biased.

- Reject-Biased
  #. ``2d_sketch`` -- A 2D sketch or lineart. May also indicate content otherwise incomprehensible
     to the model.
  #. ``2d_bad`` -- An inferior quality 2D illustration, or one which contains aesthetically
     unpleasing or objectionable content.
  #. ``2d_cute`` -- A childlike or highly "cute"-styled 2D depiction.

  #. ``3d_model_bad`` -- An inferior quality 3D model, or one which contains aesthetically
     unpleasing or objectionable content.
  #. ``3d_model_simple`` -- A low detail or simplified 3D model, such as a toy or game-ready asset.
  #. ``3d_scene_bad`` -- An aesthetically unpleasing 3D scene or environment. May be a result of
     disfavored content or excessive visual obstructions.
  #. ``3d_untex_bad`` -- An untextured 3D model of similar overall quality and content to
     ``3d_model_bad``.
  #. ``3d_multi`` -- Multiple redundant views of a 3D model.
  #. ``3d_untex_multi`` -- Like ``3d_multi``, but untextured.
  #. ``2d_multi`` -- A collage of multiple illustrations.
  #. ``multi_img`` -- A collage of distinct images, which in some cases may be superimposed over
     another primary image.

  #. ``3d_cute`` -- A childlike or highly "cute"-styled realistic 3D model or scene.
  #. ``3d_no_fur`` -- A reasonable-quality 3d model of an normally-furred animal that is lacking
     fur.
  #. ``3d_miniature`` -- A 3D miniature model, generally with low detail or with the characteristic
     exaggerated faux detailing.
  #. ``3d_rig`` -- A 3D model with animation rigging controls visible.
  #. ``3d_wireframe`` -- A 3D model with visible wireframe.

  #. ``text`` -- Excessive or obstructive text is present.
  #. ``real_human`` -- A plausible depiction of an actual person.
  #. ``technology`` -- Prominent science fiction theming, or real-world technological objects such
     as vehicles and industrial weaponry.
  #. ``ui`` -- Software user interface or screen capture.
  #. ``degrated`` -- Excessive image degradation, such as from lossy compression, focus issues, or
     extreme motion blur.

## Component Models
The following component models will be automatically downloaded from Huggingface or Github. Their
weights are not distributed by this repository. This repository contains minimal PyTorch
implementations of some simple components of these models for compatibility purposes only.

- [https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384] (primary backbone, Apache-2.0)
- [https://github.com/miccunifi/ARNIQA] (image quality assessment, CC-BY-NC-4.0)
- [https://github.com/discus0434/aesthetic-predictor-v2-5] (aesthetic gradient, AGPL v3)

Most of the weights of FARARMIR consist of a finetuned attention pooling module for SigLIP. The
addition of other models resulted in a small but noticeable improvement in validation performance.

## Dataset
FARAMIR was trained on a hand-labelled and carefully-reviewed dataset of 6500 publicly-available
images organized into approximately 70 classes. Validation was performed against an additional set
of approximately 1400 images. Final validation performance was an MCC of 0.78 and a CTI of 0.70.
Additional human review indicates that this performance is representative and that overall
performance is excellent.

## Training
Full training code will not be provided. If you wish to finetune FARAMIR, unfreeze the following
layers:

- ``head.linear_3``
- ``head.activation_2``
- ``head.linear_4``

Do not use weight decay on biases or on ``AdaptiveSigmoid`` parameters and start with a learning
rate around ``4e-4``.

Unfreezing other portions of the head or the attention pool will most likely result in a rapid and
catastrophic decline in validation performance.

## Additional Notes
- The assessment confidence is likely useful as an aesthetic gradient.
- The output of the gated classifier heads can be examined for a more detailed breakdown of the
  assessment. All raw model outputs are Baysean logits.
- The model attempts to detect and reject depictions of children, images of real people, and
  depictions of extreme violence, gore, horror, and pestilence, but its performance in this regard
  is not sufficiently reliable to act in a safety or censorship role.

## Citations
- [https://arxiv.org/abs/2303.15343]
- [https://arxiv.org/abs/2310.14918]
- [https://arxiv.org/abs/2405.15682]
- [https://arxiv.org/abs/2405.20768] (Inspiration only.)

## License and Attribution
Model source code and weights are released under the terms of the Mozilla Public License v2.0. This
model is non-generative.

Copyright 2024 by @RedHotTensors and released by
[Project RedRocket](https://huggingface.co/RedRocket).
