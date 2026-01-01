

::: {.cell .markdown}

## Launch a Jupyter container

Inside the SSH session, start a Jupyter container:

```bash
# run on node-eval-offline
docker run  -d --rm  -p 8888:8888 \
    -v ~/eval-offline-chi/workspace:/home/jovyan/work/ \
    -v food11:/mnt/ \
    -e FOOD11_DATA_DIR=/mnt/Food-11 \
    --name jupyter \
    quay.io/jupyter/pytorch-notebook:pytorch-2.5.1
```

To access the Jupyter service, we will need its randomly generated secret token (which secures it from unauthorized access). We'll get this token by running `jupyter server list` inside the `jupyter` container:

```bash
# run on node-eval-offline
docker exec jupyter jupyter server list
```

Look for a line like

```
http://localhost:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Paste this into a browser tab, but in place of `localhost`, substitute the floating IP assigned to your instance, to open the Jupyter notebook interface that is running *on your compute instance*.

Open a terminal inside this Jupyter container, and run

```bash
# run in Jupyter container on node-eval-offline
pip install grad-cam pytest
```

Then, in the file browser on the left side, open the "work" directory and then click on the `eval_offline.ipynb` notebook to continue.

:::
