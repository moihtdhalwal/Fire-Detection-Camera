import cv2
import numpy as np
import math

windowName = "OpenCV Video Player"
cv2.namedWindow(windowName)

cap = cv2.VideoCapture(0)

ret, lframe = cap.read()
Ll, A, B = cv2.split(lframe)
SI = np.zeros(np.shape(Ll))
print(SI)
NOl = 0
NOc = 0
CGO = 0
x = 1
dt = 0
nop = 0
count = 0
mc = 0
while (ret):
    ret, frame = cap.read()

    if ret:
        Ll, A, B = cv2.split(lframe)
        ret, frame = cap.read()
        cframe = frame
        cv2.imshow("FIRE", frame)
        b, g, r = cv2.split(frame)
        rt = 230
        gb = cv2.compare(g, b, cv2.CMP_GT)
        rg = cv2.compare(r, g, cv2.CMP_GT)
        rrt = cv2.compare(r, rt, cv2.CMP_GT)
        rgb = cv2.bitwise_and(rg, gb)
        im = cv2.bitwise_and(rgb, rrt)
        # cv2.imshow("RGB",im)

        t = 5
        p = 1
        k = cv2.getStructuringElement(cv2.MORPH_CROSS, ((t, t,)))
        # print(k)
        dil = cv2.dilate(im, k, iterations=p)
        er = cv2.erode(im, k, iterations=p)
        fin = cv2.bitwise_and(er, dil)
        # cv2.imshow("FIRE",frame)
        # cv2.imshow("segmented fire",cv2.bitwise_and(frame,frame,mask=fin))

        img_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        Y, Cr, Cb = cv2.split(img_ycrcb)
        Ym = np.mean(Y)
        Crm = np.mean(Cr)
        Cbm = np.mean(Cb)

        I1 = cv2.compare(Y, Ym, cv2.CMP_GT)
        I2 = cv2.compare(Cb, Cbm, cv2.CMP_LT)
        I3 = cv2.compare(Cr, Crm, cv2.CMP_GT)
        I12 = cv2.bitwise_and(I1, I2)
        I23 = cv2.bitwise_and(I2, I3)
        I123 = cv2.bitwise_and(I12, I23)
        # cv2.imshow("I123",I123)
        cbcrdiff = cv2.absdiff(Cb, Cr)
        asd = cv2.compare(cbcrdiff, 40, cv2.CMP_GT)
        # cv2.imshow("asd", asd)

        img_cie = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        L, a, b = cv2.split(img_cie)
        Lm = np.mean(L)
        am = np.mean(a)
        bm = np.mean(b)
        R1 = cv2.compare(L, Lm, cv2.CMP_GT)
        R2 = cv2.compare(a, am, cv2.CMP_GT)
        R3 = cv2.compare(b, bm, cv2.CMP_GT)
        R4 = cv2.compare(b, a, cv2.CMP_GT)
        R12 = cv2.bitwise_and(R1, R2)
        R34 = cv2.bitwise_and(R3, R4)
        R14 = cv2.bitwise_and(R1, R4)
        # cv2.imshow("R14",R14)

        kl = cv2.getStructuringElement(cv2.MORPH_CROSS, ((t, t,)))
        e_R14 = cv2.erode(R14, kl, iterations=p)
        d_R14 = cv2.dilate(R14, kl, iterations=p)
        bin_cie = cv2.bitwise_and(e_R14, d_R14)
        # cv2.imshow("cie_res",cv2.bitwise_and(frame,frame,mask=bin_cie))

        im = cv2.bitwise_and(bin_cie, im)
        # cv2.imshow("RGBCIE",im)

        cframe = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        Lc, A, B = cv2.split(cframe)

        DIfd = cv2.absdiff(Lc, Ll)
        #   cv2.imshow("difd",DIfd)
        u_DIfd = np.mean(DIfd)
        sd_DIfd = np.std(DIfd)

        if u_DIfd + sd_DIfd >= 10:
            Tfd = u_DIfd + sd_DIfd
        else:
            Tfd = 10

        _, FD = cv2.threshold(DIfd, Tfd, 255, cv2.THRESH_BINARY)
        # cv2.imshow("difd2",FD)
        # print(FD)

        cntr = cv2.compare(FD, 0, cv2.CMP_EQ) / 255
        SI = np.add(cntr, SI)
        # print(SI)
        MPM = cv2.bitwise_and(FD, im)
        MPM = cv2.bitwise_and(MPM, asd)
        MPM = cv2.erode(MPM, kl, iterations=p)
        # cv2.imshow("MPM", MPM)
        CF = cv2.bitwise_and(im, MPM)
        # cv2.imshow("candi", CF)

        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(CF)

        for i in range(retval):
            if i > 0:
                if stats[i][4] > 2:
                    NOc = NOc + stats[i][4]
                # print(NOc)
            if NOc > NOl:
                CGO = CGO + 1
            NOl = NOc
            NOc = 0
            fps = 24
            if x % fps == 0:
                dt = CGO * 1.0 / fps
                print(dt)
                CGO = 0
                x = 1
            x = x + 1
            # print(dt)
            lframe = cframe
            if dt > 0.3:
                count = count + 1
                if count > 10:
                    mc = mc + 1
                if mc % 15 == 1:
                    print("fire")
                    count = 0
            if dt < 0.2:
                print("not fire")
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

cv2.destroyAllWindows()
cap.release()