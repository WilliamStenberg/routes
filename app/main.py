import dashboard as dashboard
import db as db
import importfiles as importfiles

if __name__ == '__main__':
    db.setup()
    importfiles.sync()
    dashboard.run()
