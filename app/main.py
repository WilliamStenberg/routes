import dashboard as dashboard
import overview as overview
import db as db
import importfiles as importfiles

if __name__ == '__main__':
    db.setup()
    importfiles.sync()
    overview.run()
