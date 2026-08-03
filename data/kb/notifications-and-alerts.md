# Notifications and Alerts

Flowlytics can send email or Slack notifications when a report crosses a
threshold you define.

## Setting up an alert

1. Open the report.
2. Click **Add Alert**.
3. Define a condition (e.g. "metric > 1000").
4. Choose a delivery channel: email or Slack.

## Alert delivery timing

Alerts are evaluated on the same schedule as the underlying report and are
also affected by the workspace timezone setting — an alert configured for
"weekday mornings" uses the workspace timezone to determine what counts as
morning.

## Muting alerts

Alerts can be muted for up to 30 days from the alert's detail page. Muted
alerts still evaluate in the background but do not send notifications.
