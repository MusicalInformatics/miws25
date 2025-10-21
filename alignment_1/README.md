# Introduction to Symbolic Music Alignment

This folder contains an introduction to symbolic music alignment using dynamic time warping.

To run the baseline submission for the challenge, run the following commands on the terminal:

```bash
cd PATH_TO_THE_REPO
cd alignment_1
conda activate miws25
python Baseline_Alignment.py -c -i PATH_TO_DATASET -o PATH_FOR_THE_OUTPUT
```

This will generate a compressed numpy file with the results. Please upload this file in the challenge server, which is located [here](https://challenges.cp.jku.at/challenge/30/) (you need credentials to access the repository).

## Score-to-Performance Alignment Task

This task is worth 100 points.

### Report

For this task, you are required to work in teams (you can select a team on the Moodle page) of up to 4 people. Teams of 1 person are allowed, but you have to select one of the (empty) teams on Moodle. Otherwise, you will **not** be allowed to submit your report!

For the project you will have to submit a report in the form of a blog post-like Jupyter Notebook.  The task will be graded based on the submitted report.

The deadline for the complete report is on **January 21st, 2026 23:59**.

We will grade each task on 4(+1) main aspects:

1. **Technical correctness (methods/models)**: 25% (12.5 points)
    * Methods are appropriate for the problem
    * Methods were correctly applied/used
    * Implementation is correct

2. **Technical correctness (Evaluation)**: 25% (12.5 points)
    * An evaluation was conducted
    * Correct use of metrics/statistics
    * There is no data leak (correct split of the data), if the models are trained

3. **Presentation style** (structure, clarity): 40% (20 points)
    * Ideally, the report should not be only text or code, but it also should include some figures/images illustrating the methods. Imagine that you are writing a blog post for **non-experts**, and try to explain the methods in a simple and clear way, and do not be afraid to use some math! (You can use LaTeX/Markdown on Jupyter Notebooks)
    * Is the report reasonably structured into sections? (e.g., Introduction, Methods, Datasets, etc.)
    * Are the figures illustrating the results?
    * Are the citations correctly formatted,…

4. **Critical reflection**: 10% (5 points)
    * What are the limitations of the approach, what did you learn?

5. **(Bonus) Creativity**: (up to 30%/ up to 15 points)
    * You can get extra points for creative solutions! Don’t be afraid to think outside the box, even if the results do not outperform other methods!

The report for each task must include the following points (you can structure the report in any way you want, as long as these points are covered):

1. **Introduction**:
    * What the specific task is about, why is it an interesting problem (think of musical and technical issues)

2. **Description of the method/methods used**:
    * Describe why did you select your approach (how does the method address the particular musical problem).
    * A brief description of the method(s), what are the parameters of the method (and how do they relate to the problem).
    * You don't need to include a full technical description of the methods!

3. **Evaluation of the method(s)**:
    * How do you evaluate the performance of the model? (i.e., which metrics do you use to assess the performance of the model). Include both your own evaluation using the training set, as well as the performance of the model on the leaderboard!
    * The loss function used to train a model is not necessarily the best metric to compare models (e.g., a probabilistic classifier is trained to minimize the cross entropy, but the metric used to compare the models could be the accuracy or F1-score).
    * Which datasets are you using and what is the information contained in them. Which features do you use?
    * How was the method trained? (including strategies for hyperparameter selection).

4. **Discussion of the results and your own conclusions**:
    * Discuss what worked or did not work, which characteristics of the model lead to better performance.

    * Do not be afraid to conduct ablation studies to see how different parts of the model contribute to the overall performance!

### Challenge

Each team should participate at **least once** in the challenge to get a grade in the reports. The deadline for submissions is **January 20st, 2026 23:59**! The winners of the challenge will be announced during the final concert/presentations on **January 21nd**.

For this challenge, you will have to align a performance with its score, in a note-wise fashion and export your results as a compressed Numpy file. For convenience, we will provide both a training dataset consisting of performance, score and ground truth alignments in the CSV format used for [Parangonada](https://sildater.github.io/parangonada/), an interactive interface to compare and correct alignments.

You can use **any method that you want** (even if it is not one of the methods presented in this lecture).

For developing/evaluating/(and training, if you use a method that requires it), you will use the Vienna4x22 dataset, which is a dataset consisting of 4 piano pieces and 22 different performances of each piece (by different pianists). This is one of the standard datasets for analysis of expressive music performance.

For the challenge we will use an entirely different dataset!

For the challenge, your script submission should be executed in the following way:

```bash
python TeamName_Alignment.py -c -i path_to_the_data_directory -o output_directory
```

The file `TeamName_Alignment.py` should be as self-contained as possible and you can use third-party libraries that are not included in the conda environment for the course. You can use the methods defined in the class (and  available on the [GitHub repository](https://github.com/MusicalInformatics/miws25/tree/main)). Please upload a zip file with all of the files to run your submission, including the python script itself, the conda environment yaml file, any other helper files and trained model weights (if relevant) and a `README.md` file indicating how to setup and run the code.

Please follow the example in `Baseline_Alignment.py` in the `alignment` folder in the GitHub repository.
