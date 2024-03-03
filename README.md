# PARADISE: Evaluating Implicit Planning Skills of Language Models with Procedural Warnings and Tips Dataset

**Note:** This is a preliminary version of this Github repository, which will be updated with a rigorous documentation.  

This repository contains the code and resources for the paper titled "PARADISE: Evaluating Implicit Reasoning Skills of Language Models with Procedural Warnings and Tips Dataset." The paper introduces PARADISE, a dataset designed to evaluate the implicit reasoning skills of language models. The dataset focuses on procedural warnings and tips associated with achieving specific goals in how-to tutorials. The evaluation tasks include tip and warning inference, requiring models to distinguish the correct warning/tip for a given goal without access to intermediate instructions. You can also [find](https://huggingface.co/datasets/GGLab/PARADISE) PARADISE on Hugging Face.

## Repository Structure
`data-creation`: Contains the code and resources to replicate the production of PARADISE. <br>
`src`: Includes the PARADISE dataset and related training and evaluation scripts. <br>
`media`: Holds the visual content of the paper. <br>

## Tasks
### Warning Inference
**Definition:** Warning inference aims to distinguish the correct warning for a given goal. <br>
**Example:** 

Goal: Sit up Straight at a Computer <br>
(a) Remember that people can see some of your surroundings you while you chat. Be mindful of what is in the camera’s field of view. <br>
(b) Do not remain in any one position in front of a computer for too long. <br>
(c) void moving around in this pose. Any movements you make within the pose should be deliberate and slow. <br>
(d) Keep an appropriate distance between your eyes and computer screen. <br>

### Tip Inference
**Definition:** Tip inference aims to distinguish the correct tip for a given goal. <br>
**Example:** 

Goal: Avoid Oil Splatter when Frying <br>
(a) Remember to have lots of sides apart from just the barbecued food. <br>
(b) Wear clear, plastic gloves if you are going to use your hands to mix the meat. <br>
(c) Never use extra virgin olive oil to stir-fry. It has a low smoking point. <br>
(d) Wear long sleeves when you plan on frying food. <br>

## Training

Required libraries are shared within `src/requirements.txt`. Baseline models can be simply trained and evaluated with `src/train_with_accelerator.py` and `src/test.py` files following the implementation details and using the hyperparameters shared in our paper. Please refer to our paper for cost of training each model for both tasks.

## Baseline Models

To be updated.

## Citation 

To be updated upon publication.

## Acknowledgement

This work has been supported by the Scientific and Technological Research Council of Türkiye (TÜBITAK) as part of the project “Automatic Learning of Procedural Language from Natural Language Instructions for Intelligent Assistance” with the number 121C132. We also gratefully acknowledge KUIS AI Lab for providing computational support. We thank our anonymous reviewers and the members of GGLab who helped us improve this paper. We especially thank Aysha Gurbanova, Şebnem Demirtaş, and Mahmut İbrahim Deniz for their contributions to evaluating human performance on warning and tip inference tasks.
