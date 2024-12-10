# Troubleshooting Guide {#troubleshooting}

Here are solutions to potential issues you may encounter while using the
Bodo Cloud Platform.

### Notebook: 403 Forbidden

![Notebook-403-Error](../../platform2-screenshots/notebook_403_error.png#center)

This error is typically caused by an incorrect token, often occurring when a token is cached from accessing a different workspace.
There are two recommended workarounds:

First, attempt to resolve the issue by navigating back to the organization using the left menu `Back to Organization`.
Once there, re-enter the workspace.

If the problem persists, try clearing your web browser's cache and then logging in to Bodo Platform once more.

### Notebook: 404 Not Found / 502 Bad Gateway

![Notebook-404-Error](../../platform2-screenshots/notebook_404_error.png#center)

If for some reason `My Notebook` is displaying error code 404 / 502, you should try to update and then restart the notebook server
using left menu. If this does not help, please [contact us](https://bodo.ai/contact/){target="blank"} for further assistance.

### Notebook: File Save Error

![Notebook-File-Save-Error](../../platform2-screenshots/file_save_error.png#center)

If you get a file save error with message `invalid response: 413`
make sure your notebook (`.ipynb`) file is less than 16MB in size. Bodo Platform
does not support notebook files larger than 16MB in size.
To reduce file size don't print large sections of text and clear output
cells by clicking `Edit` > `Clear All Outputs` in the notebook interface.

### Account Locked Error

![Account-Locked-Error](../../platform2-screenshots/account_locked.png#center)

When you login to the platform, if you get an account locked error with message `User is locked out. To unlock user, please contact your administrators`,
this means that your account has been dormant (no login in more than 90 days). Please [contact us](https://bodo.ai/contact/){target="blank"} to unlock your account.
