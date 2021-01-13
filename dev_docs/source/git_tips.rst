.. _git_tips:

Git Tips
========

This document is intended to serve as a reference for common issues that need
to be resolved through git. It is organized such that each header describes a
problem and the body will offer a solution using git commands.

Checking Out From the Wrong Branch
----------------------------------

**Problem Statement:** You switched to a new issue, but you forgot to checkout
master again. As a result, you commit history is inconsistent with master (because
you squashed commits), or that change isn't merged and you have both changes in your
second PR.

**Solution**: You want to manually perform a rebase to drop all of the commits associated
with your other branch. This only requires a few steps.

#. Determine the commit hash of the first commit you made to your new branch.
   To figure this out you can run ``git log``.

#. Perform an interactive rebase to make decisions about all commits sense master. This is
   accomplished with the command ``git rebase -i master``. This will bring you to an interactive
   screen like the following image.

   .. figure:: ../figs/git_manual_rebase.png
    :alt: Screenshot of a manual rebase in git.

#. Find the commit that starts with the hash you found early. For all commits above this commit
   in the listing, change the command from ``pick`` to either ``d`` or ``drop``. Do not alter any
   other commits. An example change is shown in the image below with a single commit.

   .. figure:: ../figs/git_manual_rebase_drop.png
    :alt: Screenshot of a manual rebase in git after dropping unwanted commits.

#. Save your changes. If the two branches were independent, you will not need to take any further
   actions. If the two branches made dependent changes you will need to do an interactive rebase
   and manually select which changes you want. This process is very similar to merging two branches
   that cannot be resolved automatically. Follow the instructions provided by git and select
   the changes that are consistent with your desired changes.

   **Note**: If you keep a change in both this branch and your old branch, git will register both
   as active changes in the PR. Once your older PR merges however, this will no longer be a difference
   between branches and it should not appear in the list of changes.

#. If you have already pushed the branch to Github, you will need to update the remote branch. Since
   you are doing a manual rebase, this requires doing a force push.
   ``git push origin yourname/name-of-branch --force-with-lease``

Accidentally committed to master (locally) instead of a new branch
------------------------------------------------------------------

**Problem Statement**: You started a new issue and made your first commit (possible with a lot of changes),
but you forget to checkout a new branch.

**Solution**: You can still checkout a branch from master and then you can simply drop the older commit from
master.

#. Create a new branch from master ``git checkout -b yourname/name-of-branch``.

#. Return to master to remove the commit ``git checkout master``.

#. Start a manual rebase ``git rebase -i HEAD~n``, where n is the number of commits 
   that you accidentally made to master.

#. Change the command for all commits from ``pick`` to ``d`` or ``drop``.
   Once you save your results, master should now be consisted. This step is
   important because otherwise your pull request may not properly detect differences
   from master.

#. Return to your new branch to make any other necessary changes.
   ``git checkout yourname/name-of-branch``.
