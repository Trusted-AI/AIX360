Python files containing algos added for Prof weight based method.
List:
    attach_probes.py - given a layer name in terms of tensorflow operations, attaches a probe, extricates layer outputs and saves.
    train_probes.py - Trains a probe model to fit the corresponding labels given a layers probe outputs. Also evaulates and stores in files.
    cifar_process.py - Processing file used for CIFAR-10 examples.
    resnet_main.py - Trains the complex Resnet Model with about 18 Resblocks/layers (modified the original tensorflow code)
    resnet_model.py - Specified the complex Resnet Model
    resnet_target_model.py - Specifies a simpler Resnet Model with 3,5,7,9 Resblocks (options to be used within the model file) with or without 
                             sample weighing
   resnet_target_ratio.py - Trains a simple model (3,5,7,9 Resblocks as specified in resnet_target_model.py) with weights related to ratios of probe confidences and unweighted simple models confidences. 
                            Options available to switch between ProfWeight/ProfWeight with Ratio/Just training simple model without weights.
    resnet_target_ratio_eval.py - This evaluates a checkpoint from simple models trained with/without weighing on test samples.
                             and prints the test error.
    Other checkpoints and files necessary to run these are in the box folder Supplement_AIX360_ProfWeight
