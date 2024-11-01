import cv2
import numpy as np
import os
# pts1 = [[93, 1237], [2179, 1233], [2188, 3290], [79, 3269]]
# pts1 = [[1560, 24], [1577, 1071], [540, 1072], [554, 8]]
# ตั้งค่าตาราง 9x9


def clear_console():
    # Attempt to set TERM variable and use 'clear' command
    os.environ['TERM'] = 'xterm'
    if os.system('clear') != 0:
        print("\033c", end="")


def createBoard():
    board = np.zeros((350, 350, 3), np.uint8)
    board[:, :] = [0, 76, 153]  # สีของกระดาน
    for i in range(9):  # สร้างเส้นตาราง 9 เส้น (8 ช่อง)
        cv2.line(board, (int(i * 37.5 + 25), 25), (int(i * 37.5 + 25),
                 325), (0, 0, 0), 3, cv2.LINE_AA)  # เส้นแนวตั้ง
        cv2.line(board, (25, int(i * 37.5 + 25)), (325,
                 int(i * 37.5 + 25)), (0, 0, 0), 3, cv2.LINE_AA)  # เส้นแนวนอน
    return board


def adjustColorRange(lower, upper, adjustment):
    """Adjusts the color range by a percentage"""
    lower_adj = np.maximum(lower - adjustment, 0)  # ค่าต่ำสุดไม่ให้ต่ำกว่า 0
    upper_adj = np.minimum(upper + adjustment, 255)  # ค่าสูงสุดไม่ให้เกิน 255
    return lower_adj, upper_adj


cap = cv2.VideoCapture(0)
# cap = cv2.imread("/Users/porchportal2/Downloads/IMG_1141.jpg")
# pts1 = []
pts1 = [[417, 38], [1435, 41], [1417, 1027], [417, 1018]]


def selectPoint(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts1.append([x, y])
        print(f"Clicked point: ({x}, {y})")


cv2.namedWindow('frame')
cv2.setMouseCallback('frame', selectPoint)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2160)

pts2 = [[0, 0], [0, 300], [300, 300], [300, 0]]

# white_lower = np.array([0, 0, 200])
white_lower = np.array([10, 10, 200])
white_upper = np.array([180, 30, 255])
# white_upper = np.array([180, 25, 255])
# black_lower = np.array([0, 0, 0])
# black_upper = np.array([180, 255, 31])
black_lower = np.array([35, 32, 44])
black_upper = np.array([100, 100, 100])

# ค่าคลาดเคลื่อน 10%
adjustment_value = 15
autoScan = False
switch_white = False
prev_white_count = 0
prev_black_count = 0
white_collect = []
collect = []

while True:
    _, frame = cap.read()
    if len(pts1) >= 0 and autoScan:
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray_img, 100, 0.01, 10)
        # corners = np.int0(corners)

        top_left = top_right = bottom_left = bottom_right = None
        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")
        # draw red color circles on all corners
        for i in corners:
            # x, y = i.ravel()
            x = int(i[0][0])
            y = int(i[0][1])
            cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)

            if x + y < min_x + min_y:  # Smallest sum -> Top-left
                min_x, min_y = x, y
                top_left = (x, y)
            if x - y > max_x - min_y:  # Largest difference -> Top-right
                max_x, min_y = x, y
                top_right = (x, y)
                # top_right = (2224, 1145)
            if x - y < min_x - max_y:  # Smallest difference -> Bottom-left
                min_x, max_y = x, y
                bottom_left = (x, y)
            if x + y > max_x + max_y:  # Largest sum -> Bottom-right
                max_x, max_y = x, y
                bottom_right = (x, y)

        # Draw circles on identified corners
        corner_positions = [("Top Left", top_left), ("Top Right", top_right),
                            ("Bottom Left", bottom_left), ("Bottom Right", bottom_right)]
        pts1 = [top_left, top_right, bottom_right, bottom_left]
        for label, pos in corner_positions:
            if pos is not None:
                # Draw green circles
                cv2.circle(frame, pos, 20, (0, 255, 0), -1)
                cv2.putText(frame, label, (pos[0] + 10, pos[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # pts1 = [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]
    if len(pts1) == 4:
        T = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
        QR = cv2.warpPerspective(frame, T, (300, 300))

        # แปลงภาพไปยังพื้นที่สี HSV
        hsv = cv2.cvtColor(QR, cv2.COLOR_BGR2HSV)

        # ปรับค่าขีดจำกัดสี
        lower_white, upper_white = adjustColorRange(
            white_lower, white_upper, adjustment_value)
        lower_black, upper_black = adjustColorRange(
            black_lower, black_upper, adjustment_value)

        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_black = cv2.inRange(hsv, lower_black, upper_black)

        BW_white = cv2.dilate(mask_white, np.ones((5, 5)))
        BW_white = cv2.erode(BW_white, np.ones((5, 5)))
        BW_black = cv2.dilate(mask_black, np.ones((2, 2)))
        BW_black = cv2.erode(BW_black, np.ones((2, 2)))

        cnts_white, _ = cv2.findContours(
            BW_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts_black, _ = cv2.findContours(
            BW_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        board = createBoard()
        # Count valid white stone contours
        white_stone_count = len([
            cnt for cnt in cnts_white
            # Radius check for valid white stones
            if 10 < cv2.minEnclosingCircle(cnt)[1] < 20
        ])

        # Count valid black stone contours
        black_stone_count = len([
            cnt for cnt in cnts_black
            # Radius check for valid black stones
            if 10 < cv2.minEnclosingCircle(cnt)[1] < 30
        ])
        if white_stone_count > prev_white_count:
            switch_white = False  # Next turn is black
            # print("White stone placed - Black's turn")
        elif black_stone_count > prev_black_count:
            switch_white = True  # Next turn is white
            # print("Black stone placed - White's turn")

            # Update previous counts
        prev_white_count = white_stone_count
        prev_black_count = black_stone_count

        # วางเม็ดหมากสีขาว
        for cnt in cnts_white:
            (x, y), r = cv2.minEnclosingCircle(cnt)
            if 10 < r < 20:
                cell_x = round(x / 37.5) * 37.5
                cell_y = round(y / 37.5) * 37.5
                # print(f"x: {cell_x}, y: {cell_y}")
                cell_x_collect = (cell_x/37.5) - 4
                cell_y_collect = (cell_y/37.5) - 4
                # print(f"x: {cell_x_collect}, y: {cell_y_collect}")
                cv2.circle(board, (int(cell_x)+25, int(cell_y)+25),
                           int(15), (255, 255, 255), -1)
                cv2.drawContours(QR, [cnt], -1, (0, 0, 0), 2)
                pos = (cell_x_collect, cell_y_collect)
                if pos not in collect:
                    # len(white_collect) + 1, "White", pos
                    collect.append(pos)
                    white_collect.append(f"White: {pos}")
                # cv2.drawContours(QR, [cnt], -1, (0, 0, 255), 2)  # Red contour
                # if white_collect is not
                # print(f'White piece radius: {r}')

        # วางเม็ดหมากสีดำ
        for cnt in cnts_black:

            (x, y), r = cv2.minEnclosingCircle(cnt)
            if (10 < r < 30):
                cell_x = round(x / 37.5) * 37.5
                cell_y = round(y / 37.5) * 37.5
                cell_x_collect = (cell_x/37.5) - 4
                cell_y_collect = (cell_y/37.5) - 4
                cv2.circle(board, (int(cell_x)+25, int(cell_y)+25),
                           int(15), (0, 0, 0), -1)
                cv2.drawContours(QR, [cnt], -1, (255, 255, 255), 2)
                pos = (cell_x_collect, cell_y_collect)
                if pos not in collect:
                    print(pos)
                    collect.append(pos)
                    white_collect.append(f"Black: {pos}")
                # cv2.drawContours(QR, [cnt], -1, (0, 20, 255), 2)

        clear_console()
        output = "White" if switch_white else "Black"
        for turn in range(len(white_collect)):
            print(f"{turn+1}. {white_collect[turn]}")

        cv2.putText(board, f"White: {white_stone_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(board, f"Black: {black_stone_count}", (220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(board, f"Turn: {output}", (125, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow('Board', board)
        cv2.imshow('QR', QR)

    # cv2.imshow('frame', frame)
    cv2.waitKey(40)

# cap.release()
# cv2.destroyAllWindows()
