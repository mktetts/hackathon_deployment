from web3.auto import w3
import binascii

with open("/home/itsmemuthu/Documents/blockchain/node1/keystore/UTC--2022-07-13T12-28-40.336264422Z--57c1ad565fc09823043ae18bbac558e0d47134fc") as keyfile:
    encrypted_key = keyfile.read()
    private_key = w3.eth.account.decrypt(encrypted_key, 
                                             '12345')

print(binascii.b2a_hex(private_key))


# // SPDX-License-Identifier: MIT
# pragma solidity >=0.4.22 <0.9.0;


# contract UserContract 
# {
#     uint256 public userCount = 0;
#     mapping(uint256 => User) public users;
#     mapping(uint256 => Action) public actions;

#     mapping(uint256 => uint256) public ids;

#     constructor(){}
        
#     struct User
#     {
#         address account;
# 		string userId;
# 	    string userType;
# 	    string roleCode ;
# 	    string firstName;
#         string lastName;
# 	    string email;
# 	    string password;
#         string isActive; 
#         string companyCode;
#     }
#     struct Action
#     {
# 	    string createdBy;
# 	    string createdDate;
# 	    string changedBy;
# 	    string changedDate;
#     }

#     event userAdded
#     (
#         User user,
#         Action actions,
#         address account
#     );
#     function addUser(User memory _user, Action memory _action) public 
#     {
#         users[userCount] = _user;
#         actions[userCount] = _action;
#         userCount++;
#         emit userAdded(_user, _action, msg.sender);
#     }

#     event updateUsers
#     (
#         uint256 _id,
#         User _user,
#         Action _action,
#         address account

#     );
#     function updateUser(uint256 _id, User memory _user, Action memory _action) public
#     {
#         users[_id] = _user;
#         actions[_id] = _action;
        
#         emit updateUsers(_id, _user, _action, msg.sender);
#     }

#     event deleteUsers
#     (
#         string _id,
#         address account
#     );
#     function deleteUser(uint256 _id) public
#     {
#         users[_id] = users[userCount - 1];
#         actions[_id] = actions[userCount - 1];
#         userCount--;

#         emit deleteUsers(users[_id].userId, msg.sender);
#     }
# }