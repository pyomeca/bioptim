# Contributing to BiorbdOptim
All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.
We recommend going through the list of [`issues`](https://github.com/pyomeca/BiorbdOptim/issues) to find issues that interest you, preferable those tagged with `good first issue`.
You can then get your development environment setup with the following instructions.

## Forking BiorbdOptim

You will need your own fork to work on the code.
Go to the [BiorbdOptim project page](https://github.com/pyomeca/BiorbdOptim/) and hit the `Fork` button.
You will want to clone your fork to your machine:

```bash
git clone https://github.com/your-user-name/BiorbdOptim.git
```

## Creating and activating conda environment

Before starting any development, we recommend that you create an isolated development environment. 
The easiest and most efficient way (due to the numerous dependencies of BiorbdOptim) is to use an anaconda virtual environment and to create it based on the `environment.yml` file. 

- Install [miniconda](https://conda.io/miniconda.html)
- `cd` to the BiorbdOptim source directory
- Install BiorbdOptim dependencies with:

```bash
conda env create -f environment.yml
conda activate biorbd_optim
```

## Implementing new features

Before starting to implement your new awesome feature, please discuss the implementation with the code owner to prevent any clashing with some other current developments. 
It is also a good idea to check the current opened pull-request to not redo something currently being developed. 
If your feature is mentioned in the issue section of GitHub, please assign it to yourself.
Otherwise, please open a new issue explaining what you are currently working on (and assign it to yourself!).

As soon as possible, you are asked to open a pull-request (see below) with a short but descriptive name. 
To tag that a pull-request is still in development, please add `[WIP]` at the beginning of the name.
Send commits that are as small as possible; 1 to 10 lines is probably a good guess, with again short but descriptive names. 
Be aware of the review done by the maintainers, they will contain useful tips and advices that should be integrated ASAP. 
Once you have responded to a specific comment, please respond `Done!` and tag it as resolved.

Make sure you add a minimal but meaningful example of your new feature and that you create a test with numerical values for comparison.
During your development, you can create a sandbox folder in the examples folder. 
Everything in this folder will automatically be ignored by Git. 
If by accident you add a binary file in the history (by not using a sandbox), your pull-request will be rejected and you will have to produce a new pull request free from the binary file. 

When you have completed the implementation of your new feature, navigate to your pull-request in GitHub and select `Pariterre` in the `Reviewers` drop menu. 
At the same time, if you think your review is ready to be merged, remove the `[WIP]` tag in the name (otherwise, your pull-request won't be merged). 
If your pull-request is accepted, there is nothing more to do. 
If changes are required, reply to all the comments and, as stated previously, respond `Done!` and tag it as resolved. 
Be aware that sometimes the maintainer can push modifications directly to your branch, so make sure to pull before continuing your work on the feature.

## Testing your code

Adding tests is required to get your development merged to the master branch. 
Therefore, it is worth getting in the habit of writing tests ahead of time so this is never an issue.
The BiorbdOptim test suite runs automatically on GitHub Actions, once your pull request is submitted.
However, we strongly encourage running the tests prior to submitting the pull-request.
To do so, simply run the tests folder in pytest (`pytest tests`).

## Convention of coding

BiorbdOptim tries to follow as much as possible the PEP recommendations (https://www.python.org/dev/peps/). 
Unless you have good reasons to disobey, pull-requests are required to follow these recommendations. 
I won't get into details here, if you haven't yet, you should read these :) 

Black is used to enforce the code spacing. 
BiorbdOptim is linted with 120 characters max per line. 
This means that your pull-request tests on GitHub will appear to fail if black fails. 
The easiest way to make sure black is happy is to locally run this command:
```bash
black . -l120
```
If you need to install black, you can do it via conda using the conda-forge channel.


