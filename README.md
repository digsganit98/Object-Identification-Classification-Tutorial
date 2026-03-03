# Notebook execution order

1. 00_environment_setup.ipynb
2. 01_data_ingestion_and_audit.ipynb
3. 02_detection_dataset_prep.ipynb
4. 03_detection_training_cpu.ipynb
5. 04_detection_inference_and_crop_generation.ipynb
6. 05_classification_dataset_prep.ipynb
7. 06_classification_training_cpu.ipynb
8. 07_end_to_end_pipeline_demo.ipynb
9. 08_model_comparison_and_reporting.ipynb
10. 09_custom_predictions_visual_demo.ipynb

This sequence is aligned to your actual dataset folders:
- detection: `rooftop-solar-panels-object-detection`
- classification: `rooftop-solar-panels-image-classification/Faulty_solar_panel`

## What's implemented

- End-to-end CPU pipeline:
1. YOLO object identification (`solar-panel`) training/inference.
2. Panel condition classification (6 classes) training/inference.
3. Combined demo: detect panel boxes, crop, classify each crop.
4. Reporting notebook to read metrics JSON and summarize results.

Main pieces:
- Detection training/inference utils: [detection_utils.py]
- Classification training/inference utils: [classification_utils.py]
- End-to-end demo notebook: [07_end_to_end_pipeline_demo.ipynb]
- Reporting notebook: [08_model_comparison_and_reporting.ipynb]
