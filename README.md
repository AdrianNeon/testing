# testing
-
3.2
1）注册github账号
2）创建一个仓库，选择公开仓库
3) 环境配置（在IDE的terminal上）：
git config --global user.name "名字"
git config --global user.email "邮箱"

#克隆项目
git clone https://github.com/AdrianNeon/testing

#在当前项目的文件添加或更改文件
git status
git add .
git commit -m "commit信息"
git push

3.4
1）当git status 时，出现了fatal: not a git repository。代表了我的terminal不在当前的文档。需要先cd进去当前仓库的文件夹。
2）当push时，出现了Please tell me who you are。代表了IDE不知道当前用户是谁。需要先登陆，
git config --global user.name "名字"
git config --global user.email "邮箱"
之后才能继续。

3.5
GitHub是一个很实用的线上仓库。从学习角度上，GitHub的社区对学生很有用，里面有很多成熟的项目共我们学习，除此之外，对做项目的方面，学生需要团队合作做一个项目时（例：大创），每个队员都可以使用GitHub统一一个线上的仓库，每个人都可以获取最更新的项目。这个可以避免功能需要等到最后的时候拼接（经验来说，很麻烦如果没有GitHub，然后只等到所有功能完成了再拼接）。所以总的来说，GitHub对编程提供了很方便的环境。
