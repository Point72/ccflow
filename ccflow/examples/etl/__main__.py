import hydra

from ccflow.utils.hydra import cfg_run

__all__ = ("main",)


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg):
    cfg_run(cfg)


# Extract step:
# python -m ccflow.examples.etl +callable=extract +context=[]
# Change url, as example of context override:
# python -m ccflow.examples.etl +callable=extract +context=["http://lobste.rs"]
# Change file name, as example of callable override:
# python -m ccflow.examples.etl +callable=extract +context=["http://lobste.rs"] ++extract.publisher.name=lobsters

# Transform step:
# python -m ccflow.examples.etl +callable=transform +context=[]
# python -m ccflow.examples.etl +callable=transform +context=[] ++transform.model.file=lobsters.html ++transform.publisher.name=lobsters

# Load step:
# python -m ccflow.examples.etl +callable=load +context=[]
# python -m ccflow.examples.etl +callable=load +context=[] ++load.file=lobsters.csv ++load.db_file=":memory:"

# View SQLite DB:
# sqlite3 etl.db
# .tables
# select * from links;
# .quit

# [project.scripts]
# etl = "ccflow.examples.etl:main"
# etl-explain = "ccflow.examples.etl:explain"

if __name__ == "__main__":
    main()
