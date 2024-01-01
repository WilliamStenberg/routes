from dashboard import Model
import db as db
import importfiles as importfiles

if __name__ == '__main__':
    db.setup()
    importfiles.sync()
    model = Model()
    model.run_server()
