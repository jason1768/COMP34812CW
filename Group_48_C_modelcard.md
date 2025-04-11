---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/jason1768/COMP34812CW

---

# Model Card for q36172hw-m95082rl-NLI

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model that was trained to determine the logical relationship between two given sentences (natural language inference task).


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

DeBERTa is an improved version of BERT and RoBERTa that introduces disentangled attention and an enhanced mask decoder. These enhancements allow it to achieve better performance than RoBERTa on most natural language understanding tasks, using around 80GB of training data.

- **Developed by:** Haotian Wu and Ruochen Li
- **Language(s):** English
- **Model type:** Self-supervised
- **Model architecture:** Transformers
- **Finetuned from model [optional]:** deberta-v3-base

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/microsoft/deberta-v3-base
- **Paper or documentation:** https://openreview.net/pdf?id=XPZIaotutsD

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

24,432 pairs of sentences from a natural language inference (NLI) dataset. Each pair consists of a premise and a hypothesis, with labels indicating their logical relationship (entailment, contradiction, or neutral).

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      # Main Training Hyperparameters
      - learning_rate: 4.74e-5
      - random_state 42
      - train_batch_size: 64
      - eval_batch_size: 64
      - num_epochs: 10
      - weight_decay: 0.04
      - max_grad_norm: 1.0
      - lr_scheduler_type: cosine
      - warmup_ratio: 0.06        # roughly equivalent to warmup_steps / total_steps
      - evaluation_strategy: epoch
      - save_strategy: epoch
      - metric_for_best_model: eval_loss
      - load_best_model_at_end: true
      - fp16: true
      # PEFT (LoRA) Parameters
      - lora_r: 16                # Rank of low-rank adaptation matrices
      - lora_alpha: 48            # Scaling factor
      - lora_dropout: 0.13        # Dropout for LoRA layers
      - peft_task_type: seq_cls   # PEFT task type, here it's sequence classification
      - lora_bias: none           # No bias adaptation
      

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time with Optuna: 280 minutes
      - overall training time with hardcode parameter: 40 minutes
      - duration per training epoch: 4 minutes
      - model size: 714MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

The full official development set (dev.csv) released for the NLI track, consisting of 6736 premise-hypothesis pairs. This dataset was used to evaluate the final model and generate predictions for Codabench.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - F1-score: 90.85%
      - Accuracy: 90.86%

### Results

The model obtained an F1-score of 90.85% and an accuracy of 90.86%.

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 20GB,
      - GPU: P100

### Software


      - Environment: Custom runtime (Kaggle / Colab / Linux VM)
      - Transformers: 4.51.2
      - PEFT: 0.15.1
      - Datasets: 3.5.0
      - Accelerate: 1.6.0
      - PyTorch: 2.5.1+cu124
      - Tokenizers: 0.21.0
      - Huggingface Hub: 0.30.2
      - Python: 3.11
      - CUDA Toolkit: 12.4 (cu124)
      - GPU-Related Libraries:
          - cuDNN: 9.1.0.70
          - cuBLAS: 12.4.5.8
          - cuFFT: 11.2.1.3
          - cuRAND: 10.3.5.147
          - cuSOLVER: 11.6.1.9
          - cuSPARSE: 12.3.1.170
          - Triton: 3.1.0

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->


      - Inputs longer than 128 tokens (after tokenization) will be cut off, which may cause loss of important information in longer sentences.
      - The model was trained and tested only on the Blackboard's NLI dataset. It may not work well on other domains, informal language, or different languages.
      - Like many large language models, DeBERTa may carry hidden biases from the data it was pretrained on, which could affect predictions.
      - Because hyperparameters were chosen using Optuna (a Bayesian trial-and-error method), results may vary slightly between runs unless a fixed setup is used.
    

## Additional Information

<!-- Any other information that would be useful for other people to know. -->


      - The model was trained using mixed-precision (fp16), which helps save memory and speed up training on supported GPUs (e.g., T4 or A100).
      - Predictions were generated using the full development set (dev.csv), which was used as a test set in this coursework but is not the final blind test.
      - The LoRA adapter was merged into the DeBERTa model after training, so the model can run independently without extra PEFT setup.
      - The full trained model is in this link https://drive.google.com/file/d/187MRDqhL-R8TtLHdwEGhKIAdBZnTL8rL/view?usp=sharing. #TODO 让若晨试一下能不能打开这个链接
      - Training notebook: https://www.kaggle.com/code/jason1768/comp34812-cw-c
      - Demo notebook: https://www.kaggle.com/code/jason1768/comp34812-cw-c-demo
    
