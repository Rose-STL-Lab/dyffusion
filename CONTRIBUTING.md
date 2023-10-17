# How to contribute to DYffusion?
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](CODE_OF_CONDUCT.md)

DYffusion is an open source project, so all contributions and suggestions are welcome.

You can contribute in many different ways: giving ideas, answering questions, reporting bugs, proposing enhancements,
improving the documentation, fixing bugs,...

Many thanks in advance to every contributor.

In order to facilitate healthy, constructive behavior in an open and inclusive community, we all respect and abide by
our [code of conduct](CODE_OF_CONDUCT.md).

## How to work on an open Issue?
You have the list of open Issues at: https://github.com/Rose-STL-lab/dyffusion/issues

Some of them may have the label `help wanted`: that means that any contributor is welcomed!

If you would like to work on any of the open Issues:

1. Make sure it is not already assigned to someone else. You have the assignee (if any) on the top of the right column of the Issue page.

2. You can self-assign it by commenting on the Issue page with the keyword: `#self-assign`.

3. Work on your self-assigned issue and eventually create a Pull Request.

## How to create a Pull Request?

1. Fork the [repository](https://github.com/Rose-STL-lab/dyffusion) by clicking on the 'Fork' button on the repository's page. This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

    ```bash
    git clone git@github.com:<your Github handle>/dyffusion.git
    cd dyffusion
    git remote add upstream https://github.com/Rose-STL-lab/dyffusion.git
    ```

3. Create a new branch to hold your development changes:

    ```bash
    git checkout -b a-descriptive-name-for-my-changes
    ```

    **do not** work on the `main` branch.

4. Set up a development environment by running the following command in a virtual environment:

    ```bash
    pip install -e ".[dev]"
    ```

   (If dyffusion was already installed in the virtual environment, remove
   it with `pip uninstall dyffusion` before reinstalling it in editable
   mode with the `-e` flag.)

5. Develop the features on your branch.

6. Format your code. Run `black` and `ruff` so that your newly added files look nice with the following command:

    ```bash
    make style
    ```
   
7. _(Optional)_ You can also use [`pre-commit`](https://pre-commit.com/) to format your code automatically each time run `git commit`, instead of running `make style` manually. 
To do this, install `pre-commit` via `pip install pre-commit` and then run `pre-commit install` in the project's root directory to set up the hooks.
Note that if any files were formatted by `pre-commit` hooks during committing, you have to run `git commit` again .


8. Once you're happy with your contribution, add your changed files and make a commit to record your changes locally:

    ```bash
    git add -u
    git commit
    ```

    It is a good idea to sync your copy of the code with the original
    repository regularly. This way you can quickly account for changes:

    ```bash
    git fetch upstream
    git rebase upstream/main
    ```

9. Once you are satisfied, push the changes to your fork repo using:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

   Go the webpage of your fork on GitHub. Click on "Pull request" to send your to the project maintainers for review.
