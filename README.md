# Musical Informatics KV WS2025

This repository contains materials for the Winter Semester 2024 course on [Musical Informatics](https://www.jku.at/en/institut-fuer-computational-perception/lehre/alle-lehrveranstaltungen/special-topics-musical-informatics).

For more information, see the [Moodle page of the course](https://moodle.jku.at/course/view.php?id=39386) (only for registered students).

## Setup

### 1. Install Miniconda

To install Miniconda, follow these simple steps:

1. **Download the Miniconda Installer:**
   - Visit the official Miniconda download page [here](https://docs.conda.io/en/latest/miniconda.html).

   - Choose the appropriate installer for your operating system (Windows, macOS, or Linux) and architecture (64-bit or 32-bit).

2. **Install Miniconda:** Follow the instructions for your OS and architecture from the [official webpage](https://docs.anaconda.com/miniconda/miniconda-install/).

3. **Verify Installation:**

   - Open a terminal (or Command Prompt for Windows).

   - Run the following command to verify that Miniconda was installed correctly:

     ```bash
     conda --version
     ```

   You should see the version number of Conda.

For more detailed instructions, you can refer to the official [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

### 2. Clone the repository and setup the conda environment

To create setup the environment, run the following commands in the terminal:

```bash
git clone --recurse-submodules https://github.com/MusicalInformatics/miws25.git
cd miws25
conda env create -f environment.yml
```

The command above requires `git` 2.13 or later, if you have an older version of `git`, you can use the following command instead.

```bash
git clone --recursive https://github.com/MusicalInformatics/miws25.git
cd miws25
conda env create -f environment.yml
```

(you can check your `git` version by typing `git version` in the terminal).

To activate the environment in the terminal:

```bash
conda activate miws25
```

## Lecturers

- Carlos Cancino-Chacón: [carlos.cancino_chacon@jku.at](mailto:carlos.cancino_chacon@jku.at)

## License

This repository is distributed under the MIT License.