# Contributing to `bioptim`
All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.
We recommend going through the list of [`issues`](https://github.com/pyomeca/bioptim/issues) to find issues that interest you, preferable those tagged with `good first issue`.
You can then get your development environment setup with the following instructions.

## Forking `bioptim`

You will need your own fork to work on the code.
Go to the [bioptim project page](https://github.com/pyomeca/bioptim/) and hit the `Fork` button.
You will want to clone your fork to your machine:

```bash
git clone https://github.com/your-user-name/bioptim.git
```

## Creating and activating conda environment

Before starting any development, we recommend that you create an isolated development environment. 
The easiest and most efficient way (due to the numerous dependencies of `bioptim`) is to use an anaconda virtual environment and to create it based on the `environment.yml` file. 

- Install [miniconda](https://conda.io/miniconda.html)
- `cd` to the `bioptim` source directory
- Install `bioptim` dependencies with:

```bash
conda env create -f environment.yml
conda activate biorbd_optim
```

## Implementing new features

Before implementing your awesome new feature, please discuss with the code owner to prevent any clashing with some other competing developments. 
It is also a good idea to check the current opened pull-request not to redo something currently being developed. 
If your feature is mentioned in the issue section of GitHub, please assign it to yourself.
Otherwise, please open a new issue explaining what you are currently working on (and assign it to yourself!).

As soon as possible, you are asked to open a pull-request (see below) with a short but descriptive name. 
Unless that pull-request is ready to be merged, please tag it as `work in progress` by adding `[WIP]` at the beginning of the pull-request name.
If you are ready to get your PR reviewed, you can add the tag `ready to review` by adding `[RTR]`.
If you think your PR is ready for the last review, please use the tag `ready to merge` by adding `[RTM]`.
Send commits that are as small as possible; 1 to 10 lines is probably a good guess, with again short but descriptive commit names. 
Be aware of the review done by the maintainers, they will contain useful tips and advice that should be integrated ASAP. 
Once you have responded to a specific comment, please respond `Done!` and tag it as resolved.

Make sure you add a minimal but meaningful example of your new feature in the `examples` folder and that you create a test with numerical values for comparison.
If this feature changes the API, this should also be reflected in the ReadMe.
During your development, you can create a `sandbox` folder in the examples folder. 
Everything in this folder will automatically be ignored by Git. 
If by accident you add a binary file in the history file (by not using a sandbox), your pull-request will be rejected and you will have to produce a new pull-request free from the binary file. 

When you have completed the implementation of your new feature, navigate to your pull-request in GitHub and select `Pariterre` in the `Reviewers` drop menu. 
At the same time, if you think your review is ready to be merged, remove the `[WIP]` tag in the name (otherwise, your pull-request won't be merged). 
If your pull-request is accepted, there is nothing more to do, Congrats! 
If changes are required, reply to all the comments and, as stated previously, respond `Done!` and tag them as resolved. 
Be aware that sometimes the maintainer can push modifications directly to your branch, so make sure to pull before continuing your work on that branch.

## Testing your code

Adding tests are required to get your development merged to the master branch. 
Therefore, it is very good practice to get the habit of writing tests ahead of time so this is never an issue.
The `bioptim` test suite runs automatically on GitHub every time a commit is submitted.
However, we strongly encourage running tests locally prior to submitting the pull-request.
To do so, simply run the tests folder in pytest (`pytest tests`).

## Commenting

Every function, class and module should have their respective proper docstrings completed.
The docstring convention used is NumPy. 
Moreover, if your new features is available to the lay user (i.e., it changes the API), the `ReadMe.md` should be modified accordingly.

## Convention of coding

`Bioptim` tries to follow as much as possible the PEP recommendations (https://www.python.org/dev/peps/). 
Unless you have good reasons to disregard them, your pull-request is required to follow these recommendations. 
I won't get into details here, if you haven't yet, you should read them :) 

All variable names that could be plural should be written as such.

Black version 25.11 is used to enforce the code linting. 
`Bioptim` is linted with the 120-character max per line's option. 
This means that your pull-request tests on GitHub will appear to fail if black fails. 
The easiest way to make sure black is happy is to locally run this command:
```bash
black . -l120 --exclude "external/*"
```
If you need to install black, you can do it via conda using the conda-forge channel.

