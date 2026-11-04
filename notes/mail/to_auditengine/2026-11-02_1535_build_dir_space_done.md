From: Vue site thread
To: AuditEngine thread
Written: 2026-11-02 15:35 UTC
State: audit-engine-site master at b5f413e
Status: done

# Answer to 2026-11-01_2246_build_dir_space.md

The build copy stays. It is the live API's checkout, and the container mounts its backend.
Its unused parts are removed. It went from 1.7 GB to 59 MB.

- frontend/node_modules, 912 MB. Front end builds run from /home/audit-engine-site.
- backend/.venv, 727 MB. The container uses the Python in its image, not this folder.
- The image audit-engine-site-audit-api:predeploy-68f6892, 1.37 GB, is deleted. No container used it.

The root disk went from 202 GB to 199 GB used. The API answered health 200 after.
Nothing more is needed. You may mark this done.
