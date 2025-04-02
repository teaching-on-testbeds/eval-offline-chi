
::: {.cell .markdown}

# Offline evaluation of ML systems

In this tutorial, we will practice selected techniques for evaluating machine learning systems, and then monitoring them in production.

The lifecycle of a model may look something like this:

* **Training**: Initially, a model is trained on some training data
* **Testing** (offline): If training completes successfully, the model progresses to a testing - offline evaluation - stage. In this stage, it is evaluated using a held-out evaluation set not used in training, and potentially other special evaluation sets (as we'll see in this tutorial).
* **Staging**: Given satisfactory performance on the online evaluation, the model may be *packaged* as part of a service, and then this package promoted to a staging environment that mimics the "production" service but without live users. In this staging environmenmt, we can perform integration tests against the service and also load tests to evaluate the inference performance of the system.
* **Canary** (or blue/green, or other "preliminary" live environment): From the staging environment, the service can be promoted to a canary or other preliminary environment, where it gets requests from a small fraction of live users. In this environment, we are closely monitoring the service, its predictions, and the infrastructure for any signs of problems. We will try to "close the feedback loop" so that we can evaluate how effective our model is on production data, and potentially, evaluate the system on business metrics.
* **Production**: Finally, after a thorough offline and online evaluation, we may promote the model to the live production environment, where it serves most users. We will continue monitoring the system for signs of degradation or poor performance.

In this particular section, we will evaluate a model in the *offline* testing stage - when it is not yet deployed as a "live" service accepting requests from real users. 

![This tutorial focuses on the offline testing stage.](images/stages-offline.svg)

To run this experiment, you should have already created an account on Chameleon, and become part of a project. You should also have added your SSH key to the KVM@TACC site.

:::

::: {.cell .markdown}

## Experiment resources 

For this experiment, we will provision one virtual machine on KVM@TACC.

:::

::: {.cell .markdown}

## Open this experiment on Trovi


At the beginning of your GPU lease time, you will continue with the next step, in which you bring up and configure a VM instance! To begin this step, open this experiment on Trovi:

* Use this link: [Offline evaluation of ML systems](https://chameleoncloud.org/experiment/share/) on Trovi
* Then, click “Launch on Chameleon”. This will start a new Jupyter server for you, with the experiment materials already in it, including the notebok to bring up the bare metal server.


:::
