
def __get_next_entity__(self, hop: int):
        """
        获得下n跳的所有实体
        @param entity:本级实体
        @param hop:相应的跳数
        @return: 下一级的所有实体
        """
        entity = "招商局轮船股份有限公司"
        candidate = ["?a", "?b", "?c", "?d", "?e", "?f", "?g", "?h", "?i", "?j", "?k", "?l", "?m", "?n"]
        raw_node = []
        for each_hop in range(hop):
            s = candidate[2 * each_hop]
            p = candidate[2 * each_hop + 1]
            o = candidate[2 * each_hop + 2]
            raw_node += [s, p, o, "."]
        # 构造相应的sparql句子
        # 去掉最开始的头
        del raw_node[0]
        del raw_node[-1]
        raw_sql = " ".join(raw_node)
        sql = "select  * where { "  " <file:///F:/d2r-server-0.7/holder8.nt#holder_copy/" + entity + "> " + raw_sql + "  }"
        # sql = "select  * where { ?p ?o <file:///F:/d2r-server-0.7/holder8.nt#holder_copy/" + entity + ">  }"
        
        return sql

def generate_circular_holding_query(entity1: str, entity2: str) -> str:
    """
    生成检测两个实体间环形持股的SPARQL查询
    
    参数:
        entity1: 第一个实体名称（如"招商局轮船股份有限公司"）
        entity2: 第二个实体名称（如"招商银行股份有限公司"）
    
    返回:
        完整的SPARQL查询字符串
    """
    return f"""
PREFIX : <file:///F:/d2r-server-0.7/holder8.nt#holder_copy/>
SELECT (cycleBoolean(:{entity1}, :{entity2}, true, {{}}) as ?x)
WHERE {{ }}
"""


# 使用示例
if __name__ == "__main__":
    exp = __get_next_entity__(None, 3)
    print(exp)
    # 输出: select  * where { <file:///F:/d2r-server-0.7/holder8.nt#holder_copy/招商局轮船股份有限公司> ?a ?b ?c ?d ?e ?f ?g ?h ?i ?j ?k ?l ?m ?n . }

    entityA = "招商局轮船股份有限公司"
    entityB = "招商银行股份有限公司"
    
    sparql_query = generate_circular_holding_query(entityA, entityB)
    print("生成的SPARQL查询:\n")
    print(sparql_query)