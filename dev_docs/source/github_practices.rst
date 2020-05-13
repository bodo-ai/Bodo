.. _github_practices:

Github Practices
----------------------

Issues
~~~~~~


Any identified issues in Bodo that are not already tracked on GitHub need to be opened as soon as possible so we don't lose track of them.

When opening issues:

1. Double check that the issue does not already exist (maybe with different title, alternative spelling, etc.).
2. Write a description that can be understood by people other than the author.
3. It is extremely important to include a small replicating example if possible (as small as possible)
4. Assign the issue to a release Milestone
5. Assign the issue to a Project
    * At the least it should be assigned to a milestone project. For example: if assigning issue to 2020.3 milestone, also assign to Project "Bodo Release 2020.3". If no project has been created yet for that milestone, create one (see Projects below).

Pull Requests
~~~~~~~~~~~~~

Name the branches that you push to GitHub repo as ``yourname/name-of-branch``.

When opening a PR:

1. Link to any issues that it closes by including this in the description: "Closes #X" where X is the issue number.
2. Assign the PR to a release Milestone
3. Include a summary of the major changes to make the PR easier for others to understand and review. This only requires a very small amount of time but can save a lot of time for reviewers.

    * This summary can be also used for the commit message later (e.g. if the branch is squashed before merge on GitHub, it is very easy to include it in the commit message), and the commit will have a nice description that will make it easier to know what the commit did if there is a need to refer to it in the future (this is useful even for the author when going back to a commit merged several months/years ago).

4. If the PR is a draft, mark it as such using GitHub's drop down button (https://github.blog/2019-02-14-introducing-draft-pull-requests/). No need to specify that it is a draft in the PR's title.
5. If this is a PR that cleans or refactors a lot of code, it is best to make sure that no one is working on that code at the same time, and to keep these changes isolated (not mixed with other type of changes).
6. Before assigning for review, make sure that the code is easy to review by ensuring:
    1. Code quality is good, for example:
        * Clean (no debug prints, debug code, comments that don't apply anymore, dead code, etc.)
        * Refactored
        * Simple (i.e. you can't think of ways to simply the code further)
    2. There is docstring documentation for newly added classes, methods, functions, etc.
    3. There are comments inside complex algorithms for non-obvious parts
    4. Complicated algorithms are explained at a high-level
    5. Try to keep the PR focused (for example, don't address several unrelated issues in the same PR) and _small_ to facilitate review
    6. PR should not modify parts that are unrelated to the PR, especially any large amounts of code since that will waste reviewers' time and increases the chances of code conflicts with work-in-progress PRs of others.
    7. Run code formatters (black and clang-format) on new/modified code.
7. Assign reviewers when the PR is ready for review.
8. When merging PRs:
    1. Note that it is better to keep commits in the master branch atomic (as in the commit passes CI). One good reason is because "git bisect" (which is very useful to find the commit that introduced a bug that was detected days after a commit is merged) relies on this to be effective.
    2. Some PRs have many commits, many of which are very small and just fix typos, small errors, etc. This history is not useful for the master branch and makes it harder to browse the commit history on master. The easiest way to simplify the history of a PR is to squash and merge (using GitHub UI) and keep the interesting parts of the history in the commit message.
    3. Try to have a descriptive commit message that summarizes the changes made (see point 2 above).
    4. Make sure long commit messages have line breaks. GitHub I think wraps long lines automatically when displaying them, but doesn't introduce line breaks when merging and ``git log`` on the command line will show a very long line.
    5. Delete branch after merging if there is no need to keep it in the GitHub repo.

Reviews
~~~~~~~

* **All reviewers should make sure that code not only works but is up to standard** (as listed under Pull Requests above).
* Reviewers should make sure that they understand the code before approval (not necessarily all the details but it is not possible to find errors, corner cases where things could fail, etc. if the code is not understood). Feel free to ask author any questions.
* Check that there are enough tests in the PR for all the added or changed functionality (sometimes the author can miss test cases).
* Review comments/conversations should be marked as resolved by the author of the comment.
* It is good practice to mark comments/conversations as resolved when the PR author has addressed them. This also simplifies the work of later reviewers.

Projects
~~~~~~~~

We use GitHub Projects to organize our workflows and prioritize tasks (https://github.com/Bodo-inc/Bodo/projects). The idea is that we classify our GitHub issues into one or multiple categories/projects. Each of these projects allows us to organize the issues into columns and arrange them for easy visualization.

There is one project for each upcoming Bodo release, and also some for major projects or goals like customer support and tech marketing. **An issue can be part of multiple projects**.

For Release Projects, we use a GitHub template that has a column for high priority issues/tasks and one for low priority tasks.

As explained above, when someone creates an issue, they need to assign it to the release milestone project. This will automatically place the issue in the "Needs triage" column. When an issue get closed (or the linked PR gets merged) it will automatically be moved to the "Closed column".

How to create a milestone project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. On GitHub Bodo page, go to Projects, click on "New project".
2. Name the project "Bodo Release VERSION", for example: "Bodo Release 2020.3" and a description like "Bodo March Release"
3. Under "Project template", pick "Bug triage" from the dropdown.
