# #!/bin/sh
rsync -arvh --exclude-from="sync-excludes.txt" /home/liuzc/Desktop/sail/internal/ellm my:/home/aiops/liuzc
