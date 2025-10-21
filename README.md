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

### Known Issues Setting up the environment

#### Issues setting up Partitura

If your environment has issues setting up Partitura, you should install the package directly from the source:

```bash
# Activate the miws environment
conda activate miws25

# Clone the repository
git clone https://github.com/CPJKU/partitura.git
cd partitura

# Install partitura in development mode
pip install -e .
```
#### Issues with Portaudio

If you have an error message like this, you might need to install portaudio.

```bash
issue when installing pip dependencies: src/pyaudio/device_api.c:9:10: fatal error: portaudio.h: No such file or directory
          9 | #include "portaudio.h"
            |          ^~~~~~~~~~~~~
      compilation terminated.
      error: command '/usr/bin/gcc' failed with exit code 1
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for pyaudio
error: failed-wheel-build-for-install
× Failed to build installable wheels for some pyproject.toml based projects
╰─> pyaudio
failed
CondaEnvException: Pip failed
```

 You can do this as follows:

* Linux: `sudo apt-get install portaudio19-dev`
* Mac: `brew install portaudio`

#### Issues installing Fluidsynth

The python library `pyfluidsynth` requires a working installation of Fluidsynth. On MacOS and Linux, we recommend to use `conda` to install fluidsynth.

```bash
conda install -c conda-forge fluidsynth
```
This will install the Fluidsynth library itselt (the C-based command line tool), not the python bindings (`pyfluidsynth`). 

##### Issues with Fluidsynth and pyfluidsynth on Windows

On Windows, pyfluidsynth expects fluidsynth.exe to be located in `C:\tools\bin` (other users have reported that it is expected in `C:\tools\fluidsynth\bin`). You can fix the issue by

1. Get the ZIP file for your Windows version from <https://github.com/FluidSynth/fluidsynth/releases/latest>
2. Extract the contents to `C:\tools` (or wherever pyfluidsynth expects the executable to be).

##### Using Fluidsynth installed from Homebrew on MacOS

We recommend to install Fluidsynth from conda in a dedicated environemnt. If however, you want to use the system-wide Fluidsynth installed with homebrew, you might run into an `ImportError("Couldn't find the FluidSynth library.")` with `pyfluidsynth`.  Please refer to the following [link](https://stackoverflow.com/a/75339618).

## Lecturers

- Carlos Cancino-Chacón: [carlos.cancino_chacon@jku.at](mailto:carlos.cancino_chacon@jku.at)

## License

This repository is distributed under the MIT License.