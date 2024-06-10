# Download Panda-70M Dataset

Here's a detailed guide on how to download and setup the [Panda-70M](https://snap-research.github.io/Panda-70M/) dataset for running the model.
- Download the [training csv](https://drive.google.com/file/d/1jWTNGjb-hkKiPHXIbEA5CnFwjhA-Fq_Q/view?usp=sharing);
- Download the [validation csv](https://drive.google.com/file/d/1cTCaC7oJ9ZMPSax6I4ZHvUT-lqxOktrX/view?usp=sharing);
- Download the [testing csv](https://drive.google.com/file/d/1ee227tHEO-DT8AkX7y2q6-bfAtUL-yMI/view?usp=sharing);
- Move all of the above file into the *animatediff/data* folder;
- Open a terminal in the same folder (i.e., *animatediff/data*);
- Ensure to have an active and possibly stable internet connection;
- Ensure your device is not going to be shutdown;
- Launch the following commands:
```
python panda_dataset.py --directory train --csv_filename panda70m_training_2m.csv --captions_filename train/captions_training.csv
```

```
python panda_dataset.py --directory validation --csv_filename panda70m_validation.csv --captions_filename validation/captions_validation.csv
```

```
python panda_dataset.py --directory test --csv_filename panda70m_testing.csv --captions_filename test/captions_testing.csv
```

If some of the provided links has expired, please refer to the original [webpage](https://snap-research.github.io/Panda-70M/) and download the CSV files manually. If you encounter any issue, feel free to contact us by
sending an e-mail. See *contact us* section in the main [README file](https://github.com/luca-zanchetta/lightweight-text-to-video/blob/main/README.md).
