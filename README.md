# wf-ntp-cli

The WF_NTP_script.py file was copied from the [original repository](https://github.com/impact27/WF_NTP) and only a small code section was added.

This repo provides a way to run WF-NTP as a command line tool to enable large-scale processing of videos.

## Installation
In general, the installation follows the steps of the original repository.
1. Clone the repository
2. Go to the folder of the downloaded repository (```cd wf-ntp-cli```)
3. Install all packages with ```pip install .```

The code was tested on MacOs. We recommend Python version 3.7 because WF-NTP requires a wheel of scikit-image
with version <0.16 and this wheel is only directly available for Python 3.7 with MacOs M1 (ARM).

## Usage

WF_NTP_CLI can be used from the terminal as follows:

```
python Path/to/src/wf_ntp_cli.py Path/to/Video_Name.avi Path/to/Output_Directory
```
The output directory could be for instance ```Path/to/WF-NTP-installation/data```
