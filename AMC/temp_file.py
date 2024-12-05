initialize_ax_3d.draw_3d_vector(present_vector,"green")
        initialize_ax_3d.draw_3d_vector(target_vector,"black")

        #转为X-Z二维坐标
        logger.info("X-Z Rotate")
        #X-Z转变需要转换x坐标为相反数
        present_vector_xz = np.array([-present_vector[0],present_vector[2]])
        #present_vector_xz = np.array([present_vector[2],present_vector[0]])
        target_vector_xz  = np.array([-target_vector[0],target_vector[2]])
        new_vector_1 = update_vector(present_vector_xz,target_vector_xz)
        #绘制X-Z坐标上的向量
        initialize_xz.draw_2d_vector(present_vector_xz,"green")
        initialize_xz.draw_2d_vector(target_vector_xz,"black")
        initialize_xz.draw_2d_vector(new_vector_1,"red")
        #logger.info(f"new_vector_1: {new_vector_1}")
        
        #转为X-Y二维坐标
        logger.info("X-Y Rotate")
        present_vector_xy = np.array([present_vector[1],-new_vector_1[0]])
        target_vector_xy = np.array([target_vector[1],-target_vector[0]])
        new_vector_2 = update_vector(present_vector_xy,target_vector_xy)
        #绘制X-Z坐标上的向量
        initialize_xy.draw_2d_vector(present_vector_xy,"green")
        initialize_xy.draw_2d_vector(target_vector_xy,"black")
        initialize_xy.draw_2d_vector(new_vector_2,"red")

        #logger.info(f"new_vector_2: {new_vector_2}")

        present_vector = np.array([-new_vector_2[1],new_vector_2[0],new_vector_1[1]])
        
        
        initialize_ax_3d.draw_3d_vector(present_vector,"red")