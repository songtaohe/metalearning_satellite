import DataLoader20180429 as MainDataLoader


def GetDataLoader(folder = "/data/songtao/"):

	OSM_folder = [folder+'metalearning/dataset/global/boston/',folder+'metalearning/dataset/global/chicago/',folder+'metalearning/dataset/global/la/']

	loaderOSMroad = MainDataLoader.LoaderOSM(OSM_folder, subtask=2)
	loaderOSMBuilding = MainDataLoader.LoaderOSM(OSM_folder, subtask=4)

	loaderDeepGlobalLand0 = MainDataLoader.LoaderCVPRContestLandCover(folder+'/metalearning/cvprcontextDataset/landcover/land-train/', subtask=0)
	loaderDeepGlobalLand1 = MainDataLoader.LoaderCVPRContestLandCover(folder+'/metalearning/cvprcontextDataset/landcover/land-train/', subtask=1)
	loaderDeepGlobalLand2 = MainDataLoader.LoaderCVPRContestLandCover(folder+'/metalearning/cvprcontextDataset/landcover/land-train/', subtask=2)
	loaderDeepGlobalLand3 = MainDataLoader.LoaderCVPRContestLandCover(folder+'/metalearning/cvprcontextDataset/landcover/land-train/', subtask=3)
	loaderDeepGlobalLand4 = MainDataLoader.LoaderCVPRContestLandCover(folder+'/metalearning/cvprcontextDataset/landcover/land-train/', subtask=4)
	loaderDeepGlobalLand5 = MainDataLoader.LoaderCVPRContestLandCover(folder+'/metalearning/cvprcontextDataset/landcover/land-train/', subtask=5)

	loaderDeepGlobalRoadDetection = MainDataLoader.LoaderCVPRContestRoadDetection(folder+'/metalearning/cvprcontextDataset/road/train/')

	loaderDSTLs = [MainDataLoader.LoaderKaggleDSTL(folder+'/metalearning/public_dataset/dstl_ready_to_use/', subtask=i) for i in xrange(1,11)]

	loaders = [loaderOSMroad,loaderOSMBuilding,loaderDeepGlobalLand0,loaderDeepGlobalLand1,loaderDeepGlobalLand2,loaderDeepGlobalLand3,loaderDeepGlobalLand4,loaderDeepGlobalLand5,loaderDeepGlobalRoadDetection] + loaderDSTLs

	dataloader = MainDataLoader.DataLoaderMultiplyTask(loaders)	



	return dataloader