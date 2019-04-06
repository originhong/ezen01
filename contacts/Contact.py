class Contants:
    def __init__(self, name, phone, email, addr1):
        self.name = name
        self.phone = phone
        self.email = email
        self.addr1 = addr1


    def print_info(list):
        print("이름:", list.name)
        print("전화번호:", list.phone)
        print("이메일:", list.email)
        print("주소:", list.addr1)


    @staticmethod
    def set_contact():
        name = input("이름:")
        phone = input("전화번호:")
        email = input("이메일:")
        addr1 = input("주소:")
        contact = Contants(name, phone, email, addr1)
        return contact


    @staticmethod
    def get_contacts(list):
        for i in list:
            i.print_info()

    @staticmethod
    def del_contacts( list, name):
        for i , t in enumerate(list):
            if t.name == name:
                del list[i]


    @staticmethod
    def print_menu():
        print("1.연락처 입력 : ")
        print("2.연락처 출력 : ")
        print("3.연락처 삭제 : ")
        print("4.종료 : ")
        menu = input("메뉴선택:")
        return int(menu)


    @staticmethod
    def run():
         list = []
         while 1:
             menu = Contants.print_menu()
             if menu == 1:
                 t = Contants.set_contact()
                 list.append(t)
             elif menu == 2:
                 Contants.get_contacts(list)
             elif menu == 3:
                 name = input("삭제할 이름")
                 Contants.del_contacts(list, name)
             elif menu == 4:
                 break