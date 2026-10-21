"""Toy stand-in for hdps.alerts.producer.AlertProducer: the alert stream written to files instead of Kafka.

hdps's AlertVettingTask builds every packet (hdps.alerts.models.Alert, in its _maybe_kafka) and hands it to its
producer's send_async. This class is that producer with Kafka replaced by a file: each packet is Avro-encoded with
the public alert schema (hdps_alerts/schemas/*.avsc, read from the hdps checkout at run time, so a new schema merged
into hdps is used as it arrives) and written as an Avro object container file per exposure (the schema embedded).
ADR-0022: the external contract is Avro binary under argus.alert; on Kafka each message is one schemaless Avro
record, here the file's records are the same encoding. Nothing in a packet is changed or added.
"""
from pathlib import Path

# The four schema files in named-type dependency order (as hdps_alerts/tests/test_models.py loads them).
SCHEMA_FILES = ("argus.alert.candidate.avsc", "argus.alert.eventAssociation.avsc",
                "argus.alert.ssoAssociation.avsc", "argus.alert.avsc")


def schema_dir():
    """hdps_alerts/schemas in the installed hdps checkout."""
    import hdps.alerts

    pkg = Path(hdps.alerts.__file__).resolve()
    for parent in pkg.parents:
        if (parent / "schemas" / "argus.alert.avsc").exists():
            return parent / "schemas"
    raise FileNotFoundError(f"hdps_alerts/schemas not found above {pkg}")


def alert_schema():
    """The argus.alert envelope from hdps's .avsc files, its named records inlined (so the container file's
    embedded schema is complete and any Avro reader can decode it)."""
    from fastavro.schema import load_schema_ordered

    d = schema_dir()
    return load_schema_ordered([str(d / name) for name in SCHEMA_FILES])


class AvroFileProducer:
    """AlertProducer's send_async, collecting packets; write(path) writes them as one Avro container file."""

    def __init__(self):
        self.schema = alert_schema()
        self.pending = []

    def send_async(self, alert, topic=None):
        self.pending.append(alert.model_dump())  # the record hdps's schema round-trip test encodes

    def write(self, path):
        """Write the collected packets to path (an Avro object container file); returns the count."""
        from fastavro import writer

        n = len(self.pending)
        if n:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                writer(f, self.schema, self.pending, codec="deflate")
        self.pending = []
        return n


def read(path):
    """The packets of an alert file, as dictionaries (fastavro reader against the embedded schema)."""
    from fastavro import reader

    with open(path, "rb") as f:
        return list(reader(f))
