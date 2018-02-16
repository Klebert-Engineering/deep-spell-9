# (C) 2017 Klebert Engineering GmbH

# =============================[ Imports ]===========================

import sqlite3
import os
from collections import defaultdict


# =====================[ DSFtsDatabaseConnection ]===================

class DSFtsDatabaseConnection:
    """
    DSFtsDatabaseConnection represents a connection to an FTS5 NDS database.
    It may be used to lookup geographic name entries for combinations of
    {"ROAD":"..","CITY":"..","COUNTRY":"..","STATE":".."} selectors.
    """

    def __init__(self, **configuration):
        self.path = configuration.pop("path", "corpora/RoadFTS5_USA.nds")
        assert os.path.exists(self.path) and os.path.splitext(self.path)[1].lower() == ".nds"
        self.fts_table_name = configuration.pop("fts_table_name", "nameFtsTable")
        self.docid_column_name = configuration.pop("docid_column_name", "namedObjectId")
        self.morton_column_name = configuration.pop("morton_column_name", "mortonCode")
        self.class_specificity = configuration.pop("class_specificity", ["COUNTRY", "STATE", "CITY", "ROAD"])
        self.training_data_view_name = configuration.pop("training_data_view_name", "training_data")
        self.column_name_for_class = defaultdict(lambda: "criterionH", **configuration.pop("column_names", {
            "ROAD": "criterionA",
            "CITY": "criterionB",
            "STATE": "criterionC",
            "COUNTRY": "criterionD"
        }))
        self.conn = sqlite3.connect(self.path)

    def __enter__(self):
        return self.conn.execute("select * from `{training_data_view_name}`".format(
            training_data_view_name=self.training_data_view_name))

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def lookup_fts_entries(self, limit=10, **criteria):
        keys = ["docid", "morton", "count"] + list(criteria.keys())
        statement = """
        select
            `{docid_column_name}` as docid,
            `{morton_column_name}` as morton,
            count({most_specific_criterion}) as group_size,
            {criteria_aliases}
        from
            `{fts_table_name}`
        where
            `{fts_table_name}` match '{criteria_expression}'
        group by
            {most_specific_criterion}
        order by
            {criteria_length_sum},
            group_size
        limit
            {limit}
        """.format(
            docid_column_name=self.docid_column_name,
            morton_column_name=self.morton_column_name,
            most_specific_criterion=self.column_name_for_class[
                max(criteria.keys(), key=lambda cl: self.class_specificity.index(cl))],
            criteria_aliases=", ".join(
                "{} as {}".format(self.column_name_for_class[criterion], criterion)
                for criterion in criteria),
            fts_table_name=self.fts_table_name,
            criteria_expression=" ".join(
                "{}: \"{}\"".format(self.column_name_for_class[criterion], value)
                for criterion, value in criteria.items()),
            criteria_length_sum="+".join(
                "length({})".format(self.column_name_for_class[criterion])
                for criterion in criteria),
            limit=limit
        )
        print(statement)
        result = self.conn.execute(statement)
        return [dict(zip(keys, record)) for record in result.fetchall()]


if __name__ == "__main__":
    fts = DSFtsDatabaseConnection()
    print(fts.lookup_fts_entries(10, CITY="Los Angeles"))
    print(fts.lookup_fts_entries(10, STATE="California"))
    print(fts.lookup_fts_entries(10, CITY="Los Angeles", STATE="California"))
