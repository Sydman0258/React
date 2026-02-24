const mongoose=require("mongoose");

const employeeSchema=new mongoose.Schema({

name:String,
email:String,
department:String,
salary:Number,
joiningDate:Date

})

module.exports=mongoose.model("Employee",employeeSchema);