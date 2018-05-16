#!/usr/bin/python

import sys, getopt, os
from dl.com.utils.cv import cvDlTrain

def main(argv):

    runmode = "run"
    person_name = ""

    try:
        opts, args = getopt.getopt(argv, "hrunmode:person_name", ["runmode=", "person_name="])
        print(args)
        if opts:
            for o, a in opts:
                if o == "--runmode":
                    runmode = a

                if o == "--person_name":
                    person_name = a

        cd = cvDlTrain()

        if runmode =="run":
            cd.trainModel()
            cd.recognisePeople()
        else:
            cd.train(person_name)



    except getopt.GetoptError:
        print(' python run.py --runmode=train --person_name=rajan')
        sys.exit(2)


if __name__ == "__main__":
   main(sys.argv[1:])

