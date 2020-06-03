# Contributing to BiorbdOptim
All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.
We recommend going through the [`issues`](https://github.com/pyomeca/BiorbdOptim/issues) to find issues that interest you, preferable those tagged with `good first issue`.
Then, you can get your development environment setup with the following instructions.

## Forking BiorbdOptim

You will need your own fork to work on the code.
Go to the [BiorbdOptim project page](https://github.com/pyomeca/BiorbdOptim/) and hit the `Fork` button.
You will want to clone your fork to your machine:

```bash
git clone https://github.com/your-user-name/BiorbdOptim.git
```

## Creating and activating conda environment

Before starting any development, we recommend that you create an isolated development environment. 
The easiest and most efficient way (due to the numerous dependencies of BiorbdOptim) is to use an anaconda virtual environment. 

- Install [miniconda](https://conda.io/miniconda.html)
- `cd` to the BiorbdOptim source directory
- Install BiorbdOptim dependencies with:

```bash
conda env create -f environment.yml
conda activate biorbd_optim
```

## Implementing new features

Before starting to implement your new awesome feature, please discuss the implementation with the code owner so it doesn't clash with some other current developments. 

As soon as possible, you are very welcome to open a pull-request (see below) with a short but descriptive name. 
To tag that a pull-request is still in development, please add `[WIP]` at the beginning of the name.
Send as small commits as possible; 1 to 10 lines is probably a good guess, with again short but descriptive names. 
Be aware of the review done by the maintainers, they will contain useful tips and advices that should be integrated soon. 
Once you have responded to a specific comment, please respond `Done!` and tag it as resolved.

Make sure you add a minimal but meaningful example of your new feature and that you create a test with numerical values to compare with. 

When you have completed the implementation of the new feature, 

## Testing your code

Adding tests is required if you add or modify existing codes in pyomeca.
Therefore, it is worth getting in the habit of writing tests ahead of time so this is never an issue.
The pyomeca test suite runs automatically on GitHub Actions, once your pull request is submitted.
However, we strongly encourage running the tests prior to submitting the pull request.
To do so, simply run `make test`.

## Linting your code

Pyomeca uses several tools to ensure a consistent code format throughout the project.
The easiest way to use them is to run `make lint` from the source directory.

## Making the pull-request

When you want your changes to appear publicly on your GitHub page, push your forked feature branch’s commits:

```bash
git push origin new-feature
```

If everything looks good, you are ready to make a pull request.
This pull request and its associated changes will eventually be committed to the master branch and available in the next release.

1. Navigate to your repository on GitHub
2. Click on the `Pull Request` button
3. You can then click on `Files Changed` to make sure everything looks OK
4. Write a description of your changes in the Discussion tab
5. Click `Send Pull Request`

This request then goes to the repository maintainers, and they will review the code.
If you need to make more changes, you can make them in your branch, add them to a new commit and push them to GitHub.
The pull request will be automatically updated.

!!! info "PR Checklist"

    Let's summarize the steps needed to get your PR ready for submission.

    1. **Use an isolated Python environment**.

    2. **Properly test your code**. Write new tests if needed and make sure that your modification didn't break anything by running `make test`.

    3. **Properly format your code**. You can verify that it passes the formatting guidelines by running `make lint`.

    4. **Push your code and create a PR**.

    5. **Properly describe your modifications** with a helpful title and description. If this addresses an issue, please reference it.

