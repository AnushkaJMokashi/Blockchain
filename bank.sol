// SPDX-License-Identifier: GPL-3.0

pragma solidity ^0.8.26;
 contract Demo {
 uint private newbal = 3500;
 function deposit(uint x) public {
 newbal += x;
 }
 function withdraw(uint x) public {
 if (newbal < x) {
    require(newbal > x,"Insufficient Balance");
    revert("Insufficient balance");
 }
 newbal-= x;
 }
 function show() public view returns (uint) {
 return newbal;
 }
 }
