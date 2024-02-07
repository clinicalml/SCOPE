# SCOPE: Joint attention-based event prediction and longitudinal modeling in newly diagnosed and relapsed multiple myeloma

This repository contains the code of the SCOPE model and instructions for how to use the model on your own dataset. Below, we provide an example with a synthetic dataset.

### Importing your dataset

You should adapt the function `load_data()` in `mm79/data/CustomDataModule.py` so that it loads your data.

The model takes in different Tensors.

- X: the longitudinal data
- B: the static data
- R: Complementary static data, e.g. RNA-seq data (You can disable it's usage using `--use_rna=False`)
- T: treatment data (this corresponds to A in the paper)

You can test if your have implemented the class functions correctly by running:

`python CustomDataModule.py`

### Running SCOPE

Training SCOPE happens in two steps. First, we pre-train the model on the longitudinal forecasting objective. Second, we fine-tune the prediction heads on the event-prediction tasks.

An end-to-end example is provided in the notebook `example.ipynb`.

## Pre-training

Configure your sweep. Example in `example_sweep.yaml`

Run the sweep:

`python run_sweep.py --config_name=example_sweep`

Write down your sweep number.

## Fine-tuning on event-prediction tasks

### Selecting which runs to fine-tune

You can evaluate the performance of your different models by first running:

`python precompute_AR_results.py`

Make sure to change the sweep name to match the pre-trained sweep name in the python file. We have found that `precompute_AR_results` can at times fail in the middle of the processing (due to loading many different models). We have built some redundancy in this script so if it fails, you can simply run it again and it will resume from where it left.

Compute your pre-training results using the example notebook `example.ipynb`. 

This will output a list of pre-trained models you want to fine-tune.

### Fine-tuning

Configure your sweep in `fine_tune_example_sweep.yaml`. Write down the runs that need to be fine-tuned (from the last step) in the yaml file. Make sure to write the pretrained_sweep_id.

Then, run

`python run_sweep.py example_fine_tune_sweep.yaml`

## Evaluate

For the final evaluation of the model, you can run 

`python precompute_AR_results.py` 

While making sure you set the name of the fine-tuning sweep now.

Then run the cells in the `example.ipynb` notebook to collect the final results!

