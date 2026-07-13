from _bootstrap import add_src_to_path


add_src_to_path()

from iop.scripts.prepare_data import main


if __name__ == "__main__":
    main()
