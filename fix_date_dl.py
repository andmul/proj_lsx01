# The user wants to check the date from the first 10 rows OF THE SERVER file.
# But downloading a 100MB file just to check the date is terrible.
# Using FTP/SFTP we can read the first few bytes.

# If we are using `ftplib`, we can use `retrlines` with a callback that raises an exception to abort early.
