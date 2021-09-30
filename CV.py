#okay time to make it all as one, in a nice big loop.
begin_time = datetime.now()
#manually curated data from unseen dataset.
dataPath = "PairedimageTranslation/UnseenA_ISC/"

#predicted data from unseen dataset using the GAN
predPath = "PairedimageTranslation/converted256/out_0410_1029/"

predFilenames =[]
for file in os.listdir(predPath):
    if file.endswith(".dcm"):
        #strip the number and extension off
        predFilenames.append(file[:-5])
filenames = predFilenames

##make holding dataframe with all results for each predicted image.
Output = pd.DataFrame(columns = ['ID','Raw vs Pred MSE','Raw vs Pred SSI','Raw vs Ground truth MSE','Raw vs Ground truth SSI',
                                 'Ground truth vs Pred MSE','Ground truth vs Pred SSI','Loin Area','Bone Area','Muscle Area'
                                ,'Fat Area','Fat %','Muscle %','Bone %','Loin Perimiter','Left gigot length','Right gigot length'
                                ,'Left gigot width','Right gigot width'])


preddm = pd.DataFrame()
postdm = pd.DataFrame()
ssidm1 = []
ssidm2 = []
ssidm3 = []
msedm1 = []
msedm2 = []
msedm3 = []

###Mask Settings
T1, T2, T3, TMax = 800, 1000, 1100, 2500
#leg finding settings
    
contourVal = 300
areaVal = 300

setSize = len(filenames)

for file in filenames:
    pre = dataPath + file +"0.dcm"
    post = dataPath + file +"1.dcm"
    pred = predPath + file +"0.dcm"
    
    presample = pydicom.dcmread(pre)
    postsample = pydicom.dcmread(post)
    predsample = pydicom.dcmread(pred)
    
    prearr = presample.pixel_array
    postarr = postsample.pixel_array
    predarr = postsample.pixel_array
    
    pren, n = np.histogram(prearr.ravel(), bins=2500, range=(0,2500))
    postn, n = np.histogram(postarr.ravel(), bins=2500, range=(0,2500))
    predn, n = np.histogram(predarr.ravel(), bins=2500, range=(0,2500))
    
    ppren = pren
    ppostn = postn
    ppredn = predn
    #removing the first 500 values from the histogram to leave only tissue of interest
    ppren[:500] = 0
    ppostn[:500] = 0
    ppredn[:500] = 0
    
    #setting numpy histograms as a pandas datamatrix
    ppreddn = pd.DataFrame(np.array(ppredn))
    ppostdn = pd.DataFrame(np.array(ppostn))
    
    #histogram concatting to data matric
    preddm = pd.concat([preddm, ppreddn], axis=1, ignore_index=True)
    postdm = pd.concat([postdm, ppostdn], axis=1, ignore_index=True)
    
    # Simalarity detection

    #Image similarity 
    mse1 = mean_squared_error(prearr,predarr)
    ssim1 = ssim(prearr, predarr, data_range= predarr.max() - predarr.min())
    #print("Raw vs Predicted MSE = "+str(mse1)+" SSI = "+str(ssim1))
    
    mse2 = mean_squared_error(prearr,postarr)
    ssim2 = ssim(prearr, postarr, data_range= postarr.max() - postarr.min())
    #print("Raw vs Ground Truth MSE = "+str(mse2)+" SSI = "+str(ssim2))
    
    mse3 = mean_squared_error(postarr,predarr)
    ssim3 = ssim(postarr, predarr, data_range= predarr.max() - predarr.min())
    #print("Ground Truth vs Predicted MSE ="+str(mse3)+" SSI = "+str(ssim3))
    
    ###modify here
    msedm1.append(mse1)
    msedm2.append(mse2)
    msedm3.append(mse3)
    ssidm1.append(ssim1)
    ssidm2.append(ssim2)
    ssidm3.append(ssim3)
    
    AreaMatrix = predarr.copy()
    # Set to max were themask is true
    maskHigher =  AreaMatrix >= T1
    AreaMatrix[maskHigher] = TMax
    maskLower =  AreaMatrix < T1
    AreaMatrix[maskLower] = 0
    #plt.imshow(AreaMatrix)
    #plt.show()
    #262144 is total number of pixels
    
    #bones(dootdoot)
    BoneMatrix = predarr.copy()
    maskHigher =  BoneMatrix >= T3
    BoneMatrix[maskHigher] = TMax
    maskLower =   BoneMatrix < T3
    BoneMatrix[maskLower] = 0
    # Set to "black" (0) the pixels where mask is True
    #plt.imshow(BoneMatrix)
    #plt.show()
    
    #Muscles
    MuscleMatrix = predarr.copy()
    maskHigher =  MuscleMatrix >= T3
    MuscleMatrix[maskHigher] = 0
    maskLower =   MuscleMatrix <= T2
    MuscleMatrix[maskLower] = 0
    maskMiddle =  MuscleMatrix >= T1
    MuscleMatrix[maskMiddle] = TMax
    # Set to "black" (0) the pixels where mask is True
    #plt.imshow(MuscleMatrix)
    #plt.show()
    
    #Fat?
    FatMatrix = predarr.copy()
    maskHigher =  FatMatrix > T2
    FatMatrix[maskHigher] = 0
    maskLower =   FatMatrix < T1
    FatMatrix[maskLower] = 0
    maskMiddle =  FatMatrix >= T1
    FatMatrix[maskMiddle] = TMax
    # Set to "black" (0) the pixels where mask is True
    #plt.imshow(FatMatrix)
    #plt.show()
    
    AreaArea = (AreaMatrix > 0).sum()
    BoneArea = (BoneMatrix > 0).sum()
    MuscleArea = (MuscleMatrix > 0).sum()
    FatArea = (FatMatrix > 0).sum()
    
    AreaStr = str(AreaArea)
    
    #%Calcs
    ###values produced
    MaskPc = AreaArea/262144*100
    BonePc = BoneArea/AreaArea*100
    MusclePc = MuscleArea/AreaArea*100
    FatPc = FatArea/AreaArea*100
    
    MaskPc = "%.2f" % MaskPc
    FatPc = "%.2f" % FatPc
    MusclePc = "%.2f" % MusclePc
    BonePc = "%.2f" % BonePc
    
    ###loin perimiter
    perimeter = measure.perimeter_crofton(AreaMatrix)
    
    #time2bone
    label_img = label(BoneMatrix)
    regions = regionprops(label_img)
    
    triCords=[]
    for props in regions:
        if (props.area >=areaVal) & (props.bbox[0] <= 350):
            y0, x0 = props.centroid
            triCords += ([[x0,y0]])  
    triCords.sort(key=lambda tup: tup[1])
    triCords = triCords[:4]
    triCords.sort(key=lambda tup: tup[0], reverse = True)
    triCords = triCords[:4]
    
    #left knee
    l1 = triCords[0]
    #left ischum
    i1 = triCords[1]
    #right knee
    l2 = triCords[2]
    #right ischum
    i2 = triCords[3]
    
    #so we want distance from spine to L! and spine to L2
    ##distance USING PYTHAGORAS
    d1 = math.sqrt((abs(i1[0]-l1[0])**2) + (abs(i1[1]-l1[1])**2))
    d2 = math.sqrt((abs(i2[0]-l2[0])**2) + (abs(i2[1]-l2[1])**2))
    
    hwLx,hwLy = ((l1[0]+i1[0])/2),((l1[1]+i1[1])/2)
    hwRx,hwRy = ((l2[0]+i2[0])/2),((l2[1]+i2[1])/2)
    
    #perpendicular gradient 
    Lperp = (l1[0]-i1[0])/(l1[1]-i1[1])
    Rperp = (l2[0]-i2[0])/(l2[1]-i2[1])
    
    #equation of perpendicular bisector is
    #LeftEquation
    LXvals = range(int(i1[0]),int(l1[0]))
    LperpY = (-Lperp*(LXvals-hwLx)) + hwLy
    
    #RightEquation
    RXvals = range(int(i2[0]),int(l2[0]))
    RperpY = -Rperp*(RXvals-hwRx) + hwRy
    
    RperpY = RperpY.astype(int)
    LperpY = LperpY.astype(int)
    
    #LwidthCalc
    Lwidthpos = []
    for y,x in zip(LperpY,LXvals):
        if MuscleMatrix[y,x] > 0:
            Lwidthpos += [[x,y]]  
    LW = math.sqrt((abs(Lwidthpos[0][0]-Lwidthpos[-1][0])**2) + (abs(Lwidthpos[0][1]-Lwidthpos[-1][1])**2))
    
    #RwidthCalc
    Rwidthpos = []
    for y,x in zip(RperpY,RXvals):
        if MuscleMatrix[y,x] > 0:
            Rwidthpos += [[x,y]]  
    RW = math.sqrt((abs(Rwidthpos[0][0]-Rwidthpos[-1][0])**2) + (abs(Rwidthpos[0][1]-Rwidthpos[-1][1])**2))

    DataRow =[file[:-5],mse1,ssim1,mse2,ssim2,mse3,ssim3,float(AreaArea),float(BoneArea),float(MuscleArea)\
              ,float(FatArea),float(FatPc),float(MusclePc),float(BonePc),perimeter,d1,d2,LW,RW]
    a_series = pd. Series(DataRow, index = Output.columns)
    Output = Output.append(a_series, ignore_index=True)
    
print((datetime.now() - begin_time)/setSize)    
#print(Output)
print(Output.describe()) # gets means and stdevs

plt.figure()
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.title("Tissue Areas")
plt.bar([Output['Loin Area'][0],Output['Muscle Area'],Output['Fat Area'],Output['Bone Area']])

plt.xlabel("Pixel intensity")
plt.ylabel("Abundance")
plt.legend()
plt.subplot(1,2,2)
plt.ylim(-0.1,1.1)
plt.title("Image Set Similarities")
plt.xlabel("Mean Squared Error")
plt.ylabel("Structural Similarity Index")
imsim1=plt.scatter(msedm1,ssidm1)
imsim2=plt.scatter(msedm2,ssidm2)
imsim3=plt.scatter(msedm3,ssidm3)
plt.legend((imsim2,imsim1,imsim3),
           ('Raw vs Ground truth','Raw vs Predicted','Ground truth vs Predicted'),
           scatterpoints=1,
           ncol=1,
           fontsize=10)
plt.show()
