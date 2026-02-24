const express=require( "express");
const mongoose=require ("mongoose");
const cors=require("cors");

const app=express();

app.use(cors());
app.use(express.json());

mongoose.connect("mongodb://localhost:27017/company").then(()=>{
    console.log("mongodb connected");
})
.catch(err=>{
    console.log("Error in Db Connection !!!!");
});

const employeeRoute=require('./Routes/employeeRoutes');
app.use("/api",employeeRoute);
app.listen(5000,()=>{
    console.log("Express is running at port 5000");
});
