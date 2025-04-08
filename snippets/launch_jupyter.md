

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

Run

```bash
# run on node-eval-offline
docker logs jupyter
```

and look for a line like

```
http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Paste this into a browser tab, but in place of 127.0.0.1, substitute the floating IP assigned to your instance, to open the Jupyter notebook interface that is running *on your compute instance*.

Open a terminal inside this Jupyter container, and run

```bash
# run in Jupyter container on node-eval-offline
pip install grad-cam pytest
```

Then, in the file browser on the left side, open the "work" directory and then click on the `eval_offline.ipynb` notebook to continue.

:::

