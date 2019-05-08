def F1(path_image):

    nb_classes=len(os.listdir(path_image))
    correctly_attributed=nb_classes*[0]
    attributed=nb_classes*[0]
    nb_elements=nb_classes*[0]
    confusion_matrix= np.zeros((nb_classes,nb_classes))
    print(confusion_matrix)

    for folder in os.listdir(path_image):
        j=labels.get(folder)
        k=0
        l=0
        m=0
        n=0
        print(j)
        i = 0
        nb_elements[j] = len(os.listdir(path_image + folder))
        print(folder)
        for elements in os.listdir(path_image + folder):
            img = Image.open(path_image + folder + '/' + elements)
            img = load_image(img)/255
            y_prob = model.predict(img)
            y_classes = y_prob.argmax(axis=-1)
            attributed[y_classes[0]] = attributed[y_classes[0]]+1
            if j == y_classes[0]:
                i+=1
            if 0==y_classes[0]:
                k+=1
            elif 1==y_classes[0]:
                l+=1
            elif 2==y_classes[0]:
                m+=1
            elif 3==y_classes[0]:
                n+=1
        confusion_matrix[j,:]+=[k,l,m,n]
        correctly_attributed[j]=i

    print('nb_elements=', nb_elements )
    print('correctly_attributed=', correctly_attributed, "\n")
    print('confusion_matrix:\n',confusion_matrix, "\n")
    correctly_attributed = np.asarray(correctly_attributed)
    attributed = np.asarray(attributed)
    nb_elements = np.asarray(nb_elements)

    precision=np.asarray(correctly_attributed)/attributed
    recall=correctly_attributed/nb_elements
    F1=2*(precision*recall)/(precision+recall)

    print("Precision=", precision, '\n Recall=', recall, "\n F1=", F1)
    return
