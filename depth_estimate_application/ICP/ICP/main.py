import myICP
import dataio


def main():
    data1 = dataio.loadData("1_calib.asc")
    data2 = dataio.loadData("2_calib.asc")

    _, _, data2_ = myICP.icp(data2, data1, maxIteration = 50, tolerance = 0.00001, controlPoints = 1000)
    dataio.outputData("2_icp.asc", data2_)


if __name__ == "__main__":
    main()