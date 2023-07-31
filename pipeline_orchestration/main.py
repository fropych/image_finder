import hydra
from prefect import flow, task
from pprint import pprint


@task(retries=2)
def print_cfg(cfg):
    pprint(cfg)


@hydra.main(version_base="1.1", config_path="../configs", config_name="train.yaml")
@flow(name="Repo Info", log_prints=True)
def main(cfg):
    print_cfg(cfg)


if __name__ == "__main__":
    main()
