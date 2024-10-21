// SPDX-License-Identifier: GPL-3.0

pragma solidity >=0.8.2 <0.9.0;

contract MarksManagmtSys  
{  
    struct Student  
    {  
    int ID;  
    string fName;  
    string lName;  
    int marks;  
    }  
    address owner;  
    int public stdCount = 0;  
    mapping(int => Student) public stdRecords;  
    

    modifier onlyOwner
    {
        require(owner == msg.sender);
        _;
    }
        
     
    constructor()  
    {  
    owner=msg.sender; 
    }  

    function addNewRecords(int _ID,  string memory _fName, string memory _lName, int _marks) public onlyOwner  
    {  
    stdCount = stdCount + 1;  
    stdRecords[stdCount] = Student(_ID, _fName, _lName, _marks);  
    }  
    function bonusMarks(int _bonus) public onlyOwner  
    {  
    stdRecords[stdCount].marks = stdRecords[stdCount].marks + _bonus; 
    }

    function showAll() public view onlyOwner returns (Student[] memory) 
    {
        Student[] memory students = new Student[](uint(stdCount)); 
        for (int i = 1; i <= stdCount; i++) 
        {
            students[uint(i-1)] = stdRecords[i]; 
        }
        return students; 
    }


    function deleteRecord(int _id) public onlyOwner
    {
        require(_id > 0 && _id <= stdCount, "Invalid student ID");
        
        delete stdRecords[_id];
        stdCount = stdCount - 1; // Delete the student record

    }

}
