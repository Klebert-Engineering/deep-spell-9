# (C) 2017 Klebert Engineering GmbH

# =============================[ Imports ]===========================

import sqlite3


# =====================[ DSFtsDatabaseConnection ]===================

class DSFtsDatabaseConnection:
    """
    DSFtsDatabaseConnection represents a connection to an FTS5 NDS database.
    It may be used to lookup geographic name entries for combinations of
    {"ROAD":"..","CITY":"..","COUNTRY":"..","STATE":".."} selectors.
    """

    def __init__(self, path):
        self.conn = sqlite3.connect(path)

    def lookup_geographic_entries(self):
        pass
