# User Test Deployment Guide

This project is ready to share as a real website for class user testing.

The app is not a static export. It needs:

- the browser frontend
- the Node server in `server.mjs`
- the Python MediaPipe analysis pipeline in `backend/acting_coach`

For that reason, the safest deployment path is a Docker-based web service.

## Recommended Path

Use Render as a Docker web service.

Why this fits the current app:

- it gives you an `https://...onrender.com` URL for browser camera access
- it can run both Node and Python in one container
- it works directly with the included `Dockerfile`
- it is simple enough for a short user-test window

## What Is Already Prepared

- `Dockerfile` builds the full Node + Python app
- `render.yaml` is included for Render setup
- browser clients now send a per-device `X-Session-Id`
- uploads and generated CSV files are separated per session on the server
- the desktop OpenCV webcam route is blocked on deployed hosts and kept for local-only use

That means 20 testers will no longer overwrite each other's active CSV files during the study.

## Fastest Deployment

1. Push this folder to a GitHub repository.
2. Create a Render account and connect GitHub.
3. In Render, choose `New > Blueprint`.
4. Select the repo that contains this project.
5. Render should detect `render.yaml` automatically.
6. Create the service and wait for the first deploy to finish.
7. Open `https://your-service.onrender.com/api/health` and confirm you get JSON with `"ok": true`.
8. Open the main site URL and run one full actor-upload -> recording -> results flow.

## Instance Choice

`render.yaml` uses the `free` plan so you can get a test URL quickly.

For a real 20-person test day:

- `free` is acceptable for a lightweight pilot
- `starter` is safer if people will join around the same time

Free Render web services spin down after 15 minutes of inactivity, so the first visitor after idle time can wait about a minute for wake-up. If you stay on the free plan, open the site yourself a little before the test session starts.

## Important Test Notes

- Browser camera access needs `https://` on a remote site.
- Ask testers to use a modern Chromium browser if possible.
- The deployed site supports the browser camera flow and recorded-take upload flow.
- The original desktop Python webcam button is intentionally local-only and will not appear on the deployed site.
- Render's filesystem is ephemeral, so uploaded videos and generated CSVs are temporary. That is fine for short user tests, but not for long-term storage.

## Local Check Before You Push

```bash
docker build -t acting-coach-web .
docker run --rm -p 3001:3001 acting-coach-web
```

Then open:

```text
http://localhost:3001
```

## Environment Variables

- `HOST=0.0.0.0`
- `PORT` is supplied by Render automatically
- `PYTHON_BIN` is optional and usually not needed

## If You Want Me To Finish The Last Mile

I can also help with one of these next:

1. connect this folder to a GitHub repo layout that is ready for Render
2. add a small tester landing message / instructions screen
3. switch the deployment target from Render to Railway or another host
